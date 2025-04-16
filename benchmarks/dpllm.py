# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, ClassVar, Optional, Union, cast, overload

import cloudpickle
from typing_extensions import TypeVar, deprecated

from vllm.config import CompilationConfig
from vllm.engine.arg_utils import (EngineArgs, HfOverrides, PoolerConfig,
                                   TaskOption)
from vllm.engine.llm_engine import LLMEngine

from vllm.logger import init_logger

from vllm.sampling_params import (BeamSearchParams, GuidedDecodingParams,
                                  RequestOutputKind, SamplingParams)

from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_args, deprecate_kwargs, is_list_of
from vllm import LLM

import os
import ray
import torch
import torch_gcu

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class A:
    pass


class EngineWrapper:
    def __init__(self, engine_args) -> None:
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)

    def add_request(self, *args, **kwargs):
        self.llm_engine.add_request(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.llm_engine.step(*args, **kwargs)

    def has_unfinished_requests(self, *args, **kwargs):
        return self.llm_engine.has_unfinished_requests(*args, **kwargs)

    def get_num_unfinished_requests(self, *args, **kwargs):
        return self.llm_engine.has_unfinished_requests(*args, **kwargs)

    def execute_method(self, method, *args, **kwargs):
        return getattr(self.llm_engine, method)(*args, **kwargs)


class RayLLMWrapper:
    def __init__(self, engine_args) -> None:
        ray.init(address=None, ignore_reinit_error=True)
        num_gpus = engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size
        allowed_envs = ['ECCL_ALLTOALLV_MAXSIZE', 'TORCH_ECCL_AVOID_RECORD_STREAMS', 'ECCL_SHM_DISABLE']
        self.workers = []
        self.dp_size = int(os.environ['VLLM_DP_SIZE'])
        env_vars = {}
        for i in os.environ:
            if i.startswith('VLLM') or i in allowed_envs:
                env_vars[i] = os.environ[i]

        for rank in range(self.dp_size):
            env_vars['VLLM_DP_RANK'] = str(rank)
            gpu_ids = list(range(rank*num_gpus, (rank+1)*num_gpus))
            gpu_ids = map(lambda x: str(x % torch.gcu.device_count()), gpu_ids)
            env_vars['TOPS_VISIBLE_DEVICES'] = ','.join(gpu_ids)
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                # scheduling_strategy=scheduling_strategy,
                runtime_env={'env_vars': env_vars},
            )(EngineWrapper).remote(engine_args)
            self.workers.append(worker)

        sample_param = SamplingParams(temperature=0.0, top_p=1.0, ignore_eos=True, max_tokens=1)
        self.fake_request = {'prompt': 'hi', 'params': sample_param}
        self.request_counter = Counter()
        self.dp_idx = 0
        self.model_config = A()
        self.model_config.runner_type = "generate"
        self.model_config.supported_runner_types = ["generate"]
        self.has_unfinished = None

    def add_request(self, *args, **kwargs):
        ray.get(self.workers[self.dp_idx].add_request.remote(*args, **kwargs))
        self.dp_idx = (self.dp_idx + 1) % self.dp_size
        self.has_unfinished = None

    def step(self, *args, **kwargs):
        if self.has_unfinished is None:
            self.has_unfinished = ray.get([
                worker.has_unfinished_requests.remote()
                for worker in self.workers
            ])
        for i in range(self.dp_size):
            if not self.has_unfinished[i]:
                self.workers[i].add_request.remote(
                    request_id='fake'+str(next(self.request_counter)), **self.fake_request)

        ray_worker_outputs = ray.get([
            worker.step.remote(*args, **kwargs)
            for worker in self.workers
        ])
        results = []
        for i in range(self.dp_size):
            if self.has_unfinished[i]:
                results += ray_worker_outputs[i]
        self.has_unfinished = None

        return results

    def has_unfinished_requests(self):
        self.has_unfinished = ray.get([
            worker.has_unfinished_requests.remote()
            for worker in self.workers
        ])
        return any(self.has_unfinished)

    def get_num_unfinished_requests(self):
        ray_worker_outputs = ray.get([
            worker.get_num_unfinished_requests.remote()
            for worker in self.workers
        ])
        return sum(ray_worker_outputs)

    def shutdown(self):
        import ray
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.shutdown()


class DPLLM(LLM):

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        # After positional args are removed, move this right below `model`
        task: TaskOption = "auto",
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
        '''

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if compilation_config is not None:
            if isinstance(compilation_config, (int, dict)):
                compilation_config_instance = CompilationConfig.from_cli(
                    str(compilation_config))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = None

        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )
        assert engine_args.disable_async_output_proc, "DPLLM only supports disable_async_output_proc=True"

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = RayLLMWrapper(engine_args)
        self.engine_class = LLMEngine

        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None

    def __del__(self):
        if llm_engine := getattr(self, "llm_engine", None):
            llm_engine.shutdown()
