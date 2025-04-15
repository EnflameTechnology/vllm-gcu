import dataclasses
import gc
import inspect

import time
import weakref

from array import array
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed
import vllm.envs as envs
from torch import nn
from tqdm import tqdm
from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.abstract import AttentionState
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed import get_dp_group, get_kv_transfer_group, get_pp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    graph_capture,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata, SamplingMetadataCache
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs, MultiModalRegistry
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.prompt_adapter.worker_manager import LRUCacheWorkerPromptAdapterManager
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (
    CompletionSequenceGroupOutput,
    IntermediateTensors,
    Logprob,
    SequenceData,
    SequenceGroupMetadata,
    SequenceOutput,
    VLLM_TOKEN_ID_ARRAY_TYPE,
)
from vllm.utils import (
    DeviceMemoryProfiler,
    GiB_bytes,
    is_pin_memory_available,
    PyObjectCache,
    supports_dynamo,
)
from vllm.worker.model_runner import (
    ModelInputForGPUBuilder,
    ModelInputForGPUWithSamplingMetadata,
    TModelInputForGPU,
)
from vllm.worker.model_runner_base import InputProcessingError, ModelRunnerBase

import vllm_gcu.envs as gcu_envs

from vllm_gcu.utils import dump_memory_snapshot_when_exception


logger = init_logger(__name__)

LORA_WARMUP_RANK = 8

_NUM_WARMUP_ITERS = 2


class ModelInputForGCUBuilder(ModelInputForGPUBuilder):
    def _compute_multi_modal_input(
        self, inter_data, seq_group_metadata: SequenceGroupMetadata
    ):
        positions = inter_data.input_positions[0]
        if len(positions) == 0:
            return

        super()._compute_multi_modal_input(inter_data, seq_group_metadata)


class GCUModelRunnerBase(ModelRunnerBase[TModelInputForGPU]):
    """
    Helper class for shared methods between GCU model runners.
    """

    _model_input_cls: Type[TModelInputForGPU]
    _builder_cls: Type[ModelInputForGCUBuilder]
    builder: ModelInputForGCUBuilder

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config

        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.max_batchsize_to_capture = (
            self.vllm_config.compilation_config.max_capture_size
        )

        self.graph_runners: List[Dict[int, GCUGraphRunner]] = [
            {} for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.graph_memory_pool: Optional[Tuple[int, int]] = (
            None  # Set during graph capture.
        )

        self.has_inner_state = model_config.has_inner_state

        self.in_profile_run = False

        # When using GCU graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max seq len to capture / block size).
        self.graph_block_tables = np.zeros(
            (self.max_batchsize_to_capture, self.get_max_block_per_batch()),
            dtype=np.int32,
        )

        # Attention-free but stateful models like Mamba need a placeholder attn
        # backend, as the attention metadata is needed to manage internal state.
        # However we must bypass attention selection altogether for some models
        # used for speculative decoding to avoid a divide-by-zero in
        # model_config.get_head_size()
        num_attn_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        needs_attn_backend = num_attn_heads != 0 or self.model_config.is_attention_free

        self.attn_backend = (
            get_attn_backend(
                self.model_config.get_head_size(),
                self.model_config.dtype,
                self.kv_cache_dtype,
                self.block_size,
                self.model_config.is_attention_free,
                use_mla=self.model_config.use_mla,
            )
            if needs_attn_backend
            else None
        )
        if self.attn_backend:
            self.attn_state = self.attn_backend.get_state_cls()(weakref.proxy(self))
        else:
            self.attn_state = CommonAttentionState(weakref.proxy(self))

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry.create_input_mapper(model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
        self.prompt_adapter_manager: LRUCacheWorkerPromptAdapterManager = None

        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024**3))

        # Used to cache python objects
        self.inter_data_cache: Dict[int, PyObjectCache] = {}

        # Using the PythonizationCache in Pipeline-Parallel clobbers the
        # SequenceGroupToSample object. In Pipeline-Parallel, we have
        # more than 1 Scheduler, resulting in a potential back-to-back
        # prepare_model_inputs() call. This clobbers the cached
        # SequenceGroupToSample objects, as we reset the cache during
        # every prepare_model_inputs() call.
        self.sampling_metadata_cache: SamplingMetadataCache = (
            SamplingMetadataCache()
            if self.parallel_config.pipeline_parallel_size == 1
            else None
        )

        if hasattr(self, "_builder_cls"):
            # multi-step model runner does not have `_builder_cls`
            self.builder = self._builder_cls(weakref.proxy(self))

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler(self.device) as m:
            time_before_load = time.perf_counter()
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                assert supports_lora(
                    self.model
                ), f"{self.model.__class__.__name__} does not support LoRA yet."

                if supports_multimodal(self.model):
                    logger.warning(
                        "Regarding multimodal models, vLLM currently "
                        "only supports adding LoRA to language model."
                    )
                # It's necessary to distinguish between the max_position_embeddings
                # of VLMs and LLMs.
                if hasattr(self.model.config, "max_position_embeddings"):
                    max_pos_embeddings = self.model.config.max_position_embeddings
                else:
                    max_pos_embeddings = (
                        self.model.config.text_config.max_position_embeddings
                    )

                self.lora_manager = LRUCacheWorkerLoRAManager(
                    self.scheduler_config.max_num_seqs,
                    self.scheduler_config.max_num_batched_tokens,
                    self.vocab_size,
                    self.lora_config,
                    self.device,
                    self.model.embedding_modules,
                    self.model.embedding_padding_modules,
                    max_position_embeddings=max_pos_embeddings,
                )
                self.model = self.lora_manager.create_lora_manager(self.model)
            time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %.4f GB and %.6f seconds",
            self.model_memory_usage / float(2**30),
            (time_after_load - time_before_load),
        )

        if self.prompt_adapter_config:
            self.prompt_adapter_manager = LRUCacheWorkerPromptAdapterManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.device,
                self.prompt_adapter_config,
            )
            self.model = self.prompt_adapter_manager.create_prompt_adapter_manager(
                self.model
            )

        if (
            self.vllm_config.compilation_config.level == CompilationLevel.DYNAMO_AS_IS
            and supports_dynamo()
        ):

            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)

            if backend == "topsgraph":
                options = {"full_graph_fallback_eager": False}
            else:
                options = {}

            self.model = torch.compile(
                self.model,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend,
                options=options,
            )

    def get_model(self) -> nn.Module:
        return self.model

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from vllm.model_executor.model_loader.loader import TensorizerLoader

        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None,
    ) -> TModelInputForGPU:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If gcu graph is required, this API automatically pads inputs.
        """
        self.builder.prepare(finished_requests_ids)

        all_idle = [False] * len(seq_group_metadata_list)
        for seq_id, seq_group_metadata in enumerate(seq_group_metadata_list):
            if (
                seq_group_metadata.sampling_params.extra_args
                and seq_group_metadata.sampling_params.extra_args.get("is_idle", None)
            ):
                all_idle[seq_id] = True

        idle_seq_data = SequenceData(array(VLLM_TOKEN_ID_ARRAY_TYPE, []))
        for seq_group_metadata in seq_group_metadata_list:
            if all(all_idle):
                # handle all idle seqs
                for k, _ in seq_group_metadata.seq_data.items():
                    seq_group_metadata.seq_data[k] = idle_seq_data

            try:
                self.builder.add_seq_group(seq_group_metadata)
            except Exception as e:
                raise InputProcessingError(seq_group_metadata.request_id, str(e)) from e

        self.builder.reset_cached_inter_data()

        return self.builder.build()  # type: ignore

    @contextmanager
    def set_in_profile_run(self):
        self.in_profile_run = True
        try:
            yield
        finally:
            self.in_profile_run = False

    @torch.inference_mode()
    def profile_run(self) -> None:
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        self._dummy_run(max_num_batched_tokens, max_num_seqs)

    def _dummy_run(self, max_num_batched_tokens: int, max_num_seqs: int = 1) -> None:
        with self.set_in_profile_run():
            # Enable top-k sampling to reflect the accurate memory usage.
            sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)

            # This represents the maximum number of different requests
            # that will have unique loras, an therefore the max amount of memory
            # consumption create dummy lora request copies from the lora request
            # passed in, which contains a lora from the lora warmup path.
            dummy_lora_requests: List[LoRARequest] = []
            dummy_lora_requests_per_seq: List[LoRARequest] = []
            if self.lora_config:
                assert self.lora_manager is not None
                with self.lora_manager.dummy_lora_cache():
                    for idx in range(self.lora_config.max_loras):
                        lora_id = idx + 1
                        dummy_lora_request = LoRARequest(
                            lora_name=f"warmup_{lora_id}",
                            lora_int_id=lora_id,
                            lora_path="/not/a/real/path",
                        )
                        self.lora_manager.add_dummy_lora(
                            dummy_lora_request, rank=LORA_WARMUP_RANK
                        )
                        dummy_lora_requests.append(dummy_lora_request)
                    dummy_lora_requests_per_seq = [
                        dummy_lora_requests[idx % len(dummy_lora_requests)]
                        for idx in range(max_num_seqs)
                    ]

            # Profile memory usage with max_num_sequences sequences and the
            # total number of tokens equal to max_num_batched_tokens.
            seqs: List[SequenceGroupMetadata] = []
            # Additional GPU memory may be needed for multi-modal encoding,
            # which needs to be accounted for when calculating the GPU blocks
            # for vLLM blocker manager.
            # To exercise the worst scenario for GPU memory consumption,
            # the number of seqs (batch_size) is chosen to maximize the number
            # of images processed.

            max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
                self.model_config
            )
            if max_mm_tokens > 0:
                max_num_seqs_orig = max_num_seqs
                max_num_seqs = min(
                    max_num_seqs, max_num_batched_tokens // max_mm_tokens
                )
                if max_num_seqs < 1:
                    expr = (
                        f"min({max_num_seqs_orig}, "
                        f"{max_num_batched_tokens} // {max_mm_tokens})"
                    )
                    logger.warning(
                        "Computed max_num_seqs (%s) to be less than 1. "
                        "Setting it to the minimum value of 1.",
                        expr,
                    )
                    max_num_seqs = 1

            batch_size = 0
            for group_id in range(max_num_seqs):
                seq_len = max_num_batched_tokens // max_num_seqs + (
                    group_id < max_num_batched_tokens % max_num_seqs
                )
                batch_size += seq_len

                dummy_data = self.input_registry.dummy_data_for_profiling(
                    self.model_config, seq_len, self.mm_registry
                )

                seq = SequenceGroupMetadata(
                    request_id=str(group_id),
                    is_prompt=True,
                    seq_data={group_id: dummy_data.seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                    lora_request=(
                        dummy_lora_requests_per_seq[group_id]
                        if dummy_lora_requests_per_seq
                        else None
                    ),
                    multi_modal_data=dummy_data.multi_modal_data,
                    multi_modal_placeholders=dummy_data.multi_modal_placeholders,
                )
                seqs.append(seq)

            # Run the model with the dummy inputs.
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            # use an empty tensor instead of `None`` to force Dynamo to pass
            # it by reference, rather by specializing on the value ``None``.
            # the `dtype` argument does not matter, and we use `float32` as
            # a placeholder (it has wide hardware support).
            # it is important to create tensors inside the loop, rather than
            # multiplying the list, to avoid Dynamo from treating them as
            # tensor aliasing.
            kv_caches = [
                torch.tensor([], dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]
            finished_requests_ids = [seq.request_id for seq in seqs]
            model_input = self.prepare_model_input(
                seqs, finished_requests_ids=finished_requests_ids
            )
            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=batch_size,
                    dtype=self.model_config.dtype,
                    device=self.device,
                )

            # Disable KV Scale Calculation for dummy data during profile run
            if model_input.attn_metadata is not None:
                model_input.attn_metadata.enable_kv_scales_calculation = False

            self.execute_model(model_input, kv_caches, intermediate_tensors)
            torch.gcu.synchronize()
            if self.lora_config:
                # Remove dummy loras.
                assert self.lora_manager is not None
                self.remove_all_loras()
            return

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_adapters()

    def set_active_loras(
        self, lora_requests: Set[LoRARequest], lora_mapping: LoRAMapping
    ) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_adapters()

    def remove_all_prompt_adapters(self):
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        self.prompt_adapter_manager.remove_all_adapters()

    def set_active_prompt_adapters(
        self,
        prompt_adapter_requests: Set[PromptAdapterRequest],
        prompt_adapter_mapping: PromptAdapterMapping,
    ) -> None:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        self.prompt_adapter_manager.set_active_adapters(
            prompt_adapter_requests, prompt_adapter_mapping
        )

    def add_prompt_adapter(self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.add_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.remove_adapter(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.pin_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        if not self.prompt_adapter_manager:
            raise RuntimeError("PromptAdapter is not enabled.")
        return self.prompt_adapter_manager.list_adapters()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
        """Gcu graph capture a model.

        Note that GCU graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since GCU graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info(
            "Capturing gcugraphs for decoding. This may lead to "
            "unexpected consequences if the model is not static. To "
            "run the model in eager mode, set 'enforce_eager=True' or "
            "use '--enforce-eager' in the CLI. "
            "If out-of-memory error occurs during gcugraph capture,"
            " consider decreasing `gpu_memory_utilization` or "
            "switching to eager mode. You can also reduce the "
            "`max_num_seqs` as needed to decrease memory usage."
        )
        additional_config = self.vllm_config.additional_config
        additional_config.update({"all_dp_in_decode": True})

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.gcu.mem_get_info()[0]

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = self.max_batchsize_to_capture
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long, device=self.device)
        input_positions = torch.zeros(
            max_batch_size, dtype=torch.long, device=self.device
        )
        if self.model_config.uses_mrope:
            input_positions = torch.tile(input_positions, (3, 1)).gcu(
                device=self.device
            )
        # Prepare dummy previous_hidden_states only if needed by the model.
        # This is used by draft models such as EAGLE.
        previous_hidden_states = None
        if "previous_hidden_states" in inspect.signature(self.model.forward).parameters:
            previous_hidden_states = torch.empty(
                [max_batch_size, self.model_config.get_hidden_size()],
                dtype=self.model_config.dtype,
                device=self.device,
            )

        intermediate_inputs = None
        if not get_pp_group().is_first_rank:
            intermediate_inputs = self.model.make_empty_intermediate_tensors(
                batch_size=max_batch_size,
                dtype=self.model_config.dtype,
                device=self.device,
            )

        with self.attn_state.graph_capture(max_batch_size), graph_capture(
            self.device
        ) as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of GCU graph.
            for virtual_engine in range(self.parallel_config.pipeline_parallel_size):
                # Only rank 0 should print progress bar during capture
                gcugraph_capture_sizes = (
                    tqdm(
                        self.vllm_config.compilation_config.cudagraph_capture_sizes,
                        desc="Capturing GCU graph shapes",
                    )
                    if get_tensor_model_parallel_rank() == 0
                    else self.vllm_config.compilation_config.cudagraph_capture_sizes
                )
                for batch_size in gcugraph_capture_sizes:
                    attn_metadata = self.attn_state.graph_capture_get_metadata_for_batch(
                        batch_size,
                        is_encoder_decoder_model=self.model_config.is_encoder_decoder,
                    )
                    # Disable KV Scale Calculation for graph capture
                    attn_metadata.enable_kv_scales_calculation = False
                    if self.lora_config:
                        lora_mapping = LoRAMapping(
                            **dict(
                                index_mapping=[0] * batch_size,
                                prompt_mapping=[0] * batch_size,
                                is_prefill=False,
                            )
                        )
                        self.set_active_loras(set(), lora_mapping)

                    if self.prompt_adapter_config:
                        prompt_adapter_mapping = PromptAdapterMapping(
                            [-1] * batch_size,
                            [-1] * batch_size,
                        )
                        self.set_active_prompt_adapters(set(), prompt_adapter_mapping)
                    graph_runner = GCUGraphRunner(
                        self.model,
                        self.attn_backend.get_name(),
                        self.attn_state.graph_clone(batch_size),
                        self.model_config.is_encoder_decoder,
                    )

                    capture_inputs = {
                        "input_ids": input_tokens[:batch_size],
                        "positions": input_positions[..., :batch_size],
                        "intermediate_inputs": (
                            intermediate_inputs[:batch_size]
                            if intermediate_inputs is not None
                            else None
                        ),
                        "kv_caches": kv_caches[virtual_engine],
                        "attn_metadata": attn_metadata,
                        "memory_pool": self.graph_memory_pool,
                        "stream": graph_capture_context.stream,
                    }
                    if previous_hidden_states is not None:
                        capture_inputs["previous_hidden_states"] = (
                            previous_hidden_states[:batch_size]
                        )

                    if self.has_inner_state:
                        # Only used by Mamba-based models GCU graph atm (Jamba)
                        capture_inputs.update(
                            {
                                "seqlen_agnostic_capture_inputs": self.model.get_seqlen_agnostic_capture_inputs(
                                    batch_size
                                )
                            }
                        )
                    if self.model_config.is_encoder_decoder:
                        # add the additional inputs to capture for
                        # encoder-decoder models.
                        self._update_inputs_to_capture_for_enc_dec_model(capture_inputs)

                    with set_forward_context(
                        attn_metadata, self.vllm_config, virtual_engine
                    ):
                        graph_runner.capture(**capture_inputs)
                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.graph_runners[virtual_engine][batch_size] = graph_runner

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.gcu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        gcu_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes < 10 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            gcu_graph_size / GiB_bytes,
        )

    def _update_inputs_to_capture_for_enc_dec_model(
        self, capture_inputs: Dict[str, Any]
    ):
        """
        Updates the set of input tensors needed for GCU graph capture in an
        encoder-decoder model.

        This method modifies the provided `capture_inputs` dictionary by
        adding tensors specific to encoder-decoder specific models that
        need to be captured for GCU Graph replay.
        """
        # During the decode phase encoder_input_ids and encoder_positions are
        # unset. Do the same thing for graph capture.
        capture_inputs["encoder_input_ids"] = torch.tensor(
            [], dtype=torch.long, device=self.device
        )
        capture_inputs["encoder_positions"] = torch.tensor(
            [], dtype=torch.long, device=self.device
        )

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()


class GCUModelRunner(GCUModelRunnerBase[ModelInputForGPUWithSamplingMetadata]):
    """
    GCU model runner with sampling step.
    """

    _model_input_cls: Type[ModelInputForGPUWithSamplingMetadata] = (
        ModelInputForGPUWithSamplingMetadata
    )
    _builder_cls: Type[ModelInputForGPUBuilder] = ModelInputForGCUBuilder

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> ModelInputForGPUWithSamplingMetadata:
        # keep in line with driver
        if tensor_dict.get("input_tokens", None) is None:
            return ModelInputForGPUWithSamplingMetadata()

        model_input = ModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> ModelInputForGPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If gcu graph is required, this API automatically pads inputs.
        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids
        )
        if get_pp_group().is_last_rank:
            # Sampling metadata is only required for the final pp group
            if model_input.seq_lens is None:
                sampling_metadata = SamplingMetadata(
                    seq_groups=[],
                    selected_token_indices=torch.tensor([], device="gcu"),
                    categorized_sample_indices={
                        t: torch.tensor([]) for t in SamplingType
                    },
                    num_prompts=len(seq_group_metadata_list),
                )
                self._seq_group_metadata_list = seq_group_metadata_list
            else:
                generators = self.get_generators(finished_requests_ids)
                sampling_metadata = SamplingMetadata.prepare(
                    seq_group_metadata_list,
                    model_input.seq_lens,
                    model_input.query_lens,
                    self.device,
                    self.pin_memory,
                    generators,
                    self.sampling_metadata_cache,
                )
        else:
            sampling_metadata = None
        is_prompt = (
            seq_group_metadata_list[0].is_prompt if seq_group_metadata_list else None
        )
        return dataclasses.replace(
            model_input,
            sampling_metadata=sampling_metadata,
            is_prompt=is_prompt,
            virtual_engine=virtual_engine,
        )

    @torch.inference_mode()
    @dump_memory_snapshot_when_exception
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests, model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests, model_input.prompt_adapter_mapping
            )

        self.attn_state.begin_forward(model_input)

        # Currently gcu graph is only supported by the decode phase.
        if model_input.attn_metadata is not None:
            prefill_meta = model_input.attn_metadata.prefill_metadata
            decode_meta = model_input.attn_metadata.decode_metadata

        else:
            prefill_meta = 1  # bypass graph
            decode_meta = None

        parallel_config = self.vllm_config.parallel_config
        additional_config = self.vllm_config.additional_config
        if (
            parallel_config.enable_expert_parallel
            and parallel_config.data_parallel_size > 1
        ):
            has_prefill = torch.tensor(
                1 if model_input.attn_metadata and prefill_meta else 0,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(
                has_prefill,
                group=get_dp_group().cpu_group,
            )
            if has_prefill.item() > 0:
                # some dp rank is in prefill stage
                additional_config.update({"all_dp_in_decode": False})
                if prefill_meta is None:
                    prefill_meta = 1  # disable graph
            else:
                additional_config.update({"all_dp_in_decode": True})

        # assert model_input.attn_metadata is not None
        # prefill_meta = model_input.attn_metadata.prefill_metadata
        # decode_meta = model_input.attn_metadata.decode_metadata

        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        previous_hidden_states = kwargs.get("previous_hidden_states")
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][graph_batch_size]
            if previous_hidden_states is not None:
                previous_hidden_states = torch.cat(
                    [
                        previous_hidden_states,
                        torch.empty(
                            [
                                graph_batch_size - previous_hidden_states.shape[0],
                                *previous_hidden_states.shape[1:],
                            ],
                            dtype=previous_hidden_states.dtype,
                            device=previous_hidden_states.device,
                        ),
                    ]
                )
        else:
            model_executable = self.model

        # Receive KV cache in distributed KV cache transfer setting
        # In disagg prefill setting, it will also recv hidden states and bypass
        # model forwarding
        # In KV cache database setting, it will change the model input so that
        # we can skip prefilling on tokens that successfully received KV caches
        # NOTE: The receive operation is blocking
        bypass_model_exec = False
        if self.need_recv_kv(model_input, kv_caches):
            (
                hidden_or_intermediate_states,
                bypass_model_exec,
                model_input,
            ) = get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                # model is used to know which layer the current worker
                # is working on, so that we can receive KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches=kv_caches,
            )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = (
            {
                "finished_requests_ids": model_input.finished_requests_ids,
                "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
            }
            if self.has_inner_state
            else {}
        )
        model_kwargs = {}
        if previous_hidden_states is not None:
            model_kwargs["previous_hidden_states"] = previous_hidden_states
        if (
            self.observability_config is not None
            and self.observability_config.collect_model_forward_time
        ):
            model_forward_start = torch.gcu.Event(enable_timing=True)
            model_forward_end = torch.gcu.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            with set_forward_context(
                model_input.attn_metadata, self.vllm_config, virtual_engine
            ):
                hidden_or_intermediate_states = model_executable(
                    input_ids=(
                        model_input.input_tokens
                        if model_input.input_tokens is not None
                        else torch.tensor([], device="gcu")
                    ),
                    positions=(
                        model_input.input_positions
                        if model_input.input_positions is not None
                        else torch.tensor([], device="gcu")
                    ),
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(
                        multi_modal_kwargs, device=self.device
                    ),
                    **seqlen_agnostic_kwargs,
                    **model_kwargs,
                )

        if (
            self.observability_config is not None
            and self.observability_config.collect_model_forward_time
        ):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (
                self.is_driver_worker
                and hidden_or_intermediate_states is not None
                and isinstance(hidden_or_intermediate_states, IntermediateTensors)
                and self.observability_config is not None
                and self.observability_config.collect_model_forward_time
            ):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)
                    ).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time)
                )
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(
            hidden_or_intermediate_states, model_input.sampling_metadata
        )

        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        if gcu_envs.VLLM_GCU_SAMPLER_ON_CPU:
            logits = logits.cpu().to(torch.float32)
            selected_token_indices = (
                model_input.sampling_metadata.selected_token_indices
            )
            model_input.sampling_metadata.selected_token_indices = (
                selected_token_indices.cpu()
            )

            categorized_sample_indices = (
                model_input.sampling_metadata.categorized_sample_indices
            )
            model_input.sampling_metadata.categorized_sample_indices = {
                i: tensor.cpu() for i, tensor in categorized_sample_indices.items()
            }
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        if (
            self.observability_config is not None
            and self.observability_config.collect_model_forward_time
            and output is not None
        ):
            model_forward_end.synchronize()
            model_forward_time = model_forward_start.elapsed_time(model_forward_end)
            orig_model_forward_time = 0.0
            if intermediate_tensors is not None:
                orig_model_forward_time = intermediate_tensors.tensors.get(
                    "model_forward_time", torch.tensor(0.0)
                ).item()
            # If there are multiple workers, we are still tracking the latency
            # from the start time of the driver worker to the end time of the
            # driver worker. The model forward time will then end up covering
            # the communication time as well.
            output.model_forward_time = orig_model_forward_time + model_forward_time

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(0, indices)
                output.prefill_hidden_states = hidden_or_intermediate_states
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[: len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        if (
            model_input.input_positions is None
            and not model_input.sampling_metadata.skip_sampler_cpu_output
        ):
            output.outputs = [
                CompletionSequenceGroupOutput(
                    samples=[
                        SequenceOutput(
                            parent_seq_id=list(seq_group_metadata.seq_data.keys())[0],
                            output_token=0,
                            logprobs={0: Logprob(0.0)},
                        )
                    ],
                    prompt_logprobs=None,
                )
                for seq_group_metadata in self._seq_group_metadata_list
            ]

        return [output]

    def need_recv_kv(self, model_input, kv_caches) -> bool:
        """Check if we need to receive kv-cache from the other worker.
        We need to receive KV when
            1. current vLLM instance is KV cache consumer/decode vLLM instance
            2. this batch is not a profiling run
            3. this batch is a prefill run

        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """

        if self.vllm_config.kv_transfer_config is None:
            return False

        prefill_meta = model_input.attn_metadata.prefill_metadata

        # check if the current run is profiling
        is_profile_run = kv_caches[0].numel() == 0
        # check if the current run is prefill
        is_prefill_run = prefill_meta is not None

        return (
            self.vllm_config.kv_transfer_config.is_kv_consumer
            and (not is_profile_run)
            and is_prefill_run
        )

    def need_send_kv(self, model_input, kv_caches) -> bool:
        """Check if we need to send kv-cache to the other worker.
        We need to send KV when
            1. current vLLM instance is KV cache producer/prefill vLLM instance
            2. this batch is not a profiling run
            3. this batch is a prefill run

        Args:
            model_input: input to the model executable
            kv_caches: vLLM's paged memory
        """

        if self.vllm_config.kv_transfer_config is None:
            return False

        prefill_meta = model_input.attn_metadata.prefill_metadata

        # check if the current run is profiling
        is_profile_run = kv_caches[0].numel() == 0
        # check if the current run is prefill
        is_prefill_run = prefill_meta is not None

        return (
            self.vllm_config.kv_transfer_config.is_kv_producer
            and (not is_profile_run)
            and is_prefill_run
        )


class GCUGraphRunner(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        backend_name: str,
        attn_state: AttentionState,
        is_encoder_decoder_model: bool,
    ):
        super().__init__()
        self.model = model
        self.backend_name = backend_name
        self.attn_state = attn_state

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.gcu.GCUGraph] = None
        self._is_encoder_decoder_model = is_encoder_decoder_model

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_inputs: Optional[IntermediateTensors],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.gcu.Stream,
        **kwargs,
    ):
        assert self._graph is None
        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.compile
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_inputs,
                **kwargs,
            )
        # Wait for the warm up operations to finish before proceeding with
        # Graph Capture.
        torch.gcu.synchronize()
        # Capture the graph.
        self._graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(self._graph, pool=memory_pool, stream=stream):
            output_hidden_or_intermediate_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_inputs,
                **kwargs,
            )

            if isinstance(output_hidden_or_intermediate_states, torch.Tensor):
                hidden_or_intermediate_states = output_hidden_or_intermediate_states
            elif isinstance(output_hidden_or_intermediate_states, IntermediateTensors):
                hidden_or_intermediate_states = IntermediateTensors(
                    tensors={
                        key: value
                        for key, value in output_hidden_or_intermediate_states.tensors.items()
                    }
                )

            del output_hidden_or_intermediate_states
            # make sure `output_hidden_or_intermediate_states` is deleted
            # in the graph's memory pool
            gc.collect()
        torch.gcu.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            **self.attn_state.get_graph_input_buffers(
                attn_metadata, self._is_encoder_decoder_model
            ),
            **kwargs,
        }
        if intermediate_inputs is not None:
            self.input_buffers.update(intermediate_inputs.tensors)
        if get_pp_group().is_last_rank:
            self.output_buffers = {"hidden_states": hidden_or_intermediate_states}
        else:
            self.output_buffers = hidden_or_intermediate_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        **kwargs,
    ) -> torch.Tensor:
        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        if positions is not None:
            self.input_buffers["positions"][: positions.shape[0]].copy_(
                positions, non_blocking=True
            )

        if self.backend_name != "NO_ATTENTION":
            self.input_buffers["slot_mapping"].copy_(
                attn_metadata.slot_mapping, non_blocking=True
            )

        self.attn_state.prepare_graph_input_buffers(
            self.input_buffers, attn_metadata, self._is_encoder_decoder_model
        )

        if "seqlen_agnostic_capture_inputs" in self.input_buffers:
            self.model.copy_inputs_before_cuda_graphs(self.input_buffers, **kwargs)

        if "previous_hidden_states" in self.input_buffers:
            self.input_buffers["previous_hidden_states"].copy_(
                kwargs["previous_hidden_states"], non_blocking=True
            )

        if intermediate_tensors is not None:
            for key in intermediate_tensors.tensors:
                if key != "model_execute_time" and key != "model_forward_time":
                    self.input_buffers[key].copy_(
                        intermediate_tensors[key], non_blocking=True
                    )
        if self._is_encoder_decoder_model:
            self.input_buffers["encoder_input_ids"].copy_(
                kwargs["encoder_input_ids"], non_blocking=True
            )
            self.input_buffers["encoder_positions"].copy_(
                kwargs["encoder_positions"], non_blocking=True
            )

        # Run the graph.
        self.graph.replay()
        # Return the output tensor.
        if get_pp_group().is_last_rank:
            return self.output_buffers["hidden_states"]

        return self.output_buffers
