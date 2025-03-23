# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import torch

from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (
    CompletionSequenceGroupOutput,
    ExecuteModelRequest,
    Logprob,
    SequenceGroupMetadata,
    SequenceOutput,
)
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
from vllm.worker.model_runner_base import BroadcastableModelInput
from vllm.worker.multi_step_model_runner import (
    ModelOutput,
    MULTI_STEP_ATTENTION_BACKENDS,
    MultiStepModelRunner,
    PythonizationCache,
    StatefulModelInput,
)
from vllm.worker.worker_base import WorkerInput

from vllm_gcu.worker.model_runner import GCUModelRunnerBase
from vllm_gcu.worker.worker import GCUWorker

MULTI_STEP_ATTENTION_BACKENDS += ["xformers"]


class PatchedModelOutput(ModelOutput):
    def maybe_pythonize(self, input_metadata, copy_stream, pinned_sampled_token_buffer):
        if input_metadata.num_queries == 0:
            return
        else:
            return super().maybe_pythonize(
                input_metadata, copy_stream, pinned_sampled_token_buffer
            )


patcher = patch("vllm.worker.multi_step_model_runner.ModelOutput", PatchedModelOutput)
patcher.start()


@dataclass
class MultiStepState:
    worker_input: WorkerInput
    model_input: StatefulModelInput


class GCUMultiStepModelRunner(MultiStepModelRunner):
    def __init__(self, base_model_runner: GCUModelRunnerBase, *args, **kwargs):
        GCUModelRunnerBase.__init__(self, *args, **kwargs)

        self._base_model_runner: GCUModelRunnerBase = base_model_runner
        self.is_multi_step = self.scheduler_config.is_multi_step
        self.pinned_sampled_token_ids: Optional[torch.Tensor] = None

        self.pythonization_cache = (
            PythonizationCache()
            if self.parallel_config.pipeline_parallel_size == 1
            else None
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> StatefulModelInput:
        frozen_model_input: ModelInputForGPUWithSamplingMetadata = (
            self._base_model_runner.prepare_model_input(
                seq_group_metadata_list, virtual_engine, finished_requests_ids
            )
        )

        if (
            frozen_model_input.query_lens is None
            and frozen_model_input.seq_lens is None
            and frozen_model_input.attn_metadata is None
        ):
            self._seq_group_metadata_list = seq_group_metadata_list
            return StatefulModelInput(
                frozen_model_input=frozen_model_input,
                num_seqs=0,
                num_queries=0,
                num_single_step_prefills=0,
            )

        assert frozen_model_input.query_lens is not None
        assert frozen_model_input.seq_lens is not None
        assert frozen_model_input.attn_metadata is not None
        num_queries = len(frozen_model_input.query_lens)
        num_seqs = len(frozen_model_input.seq_lens)
        num_single_step_prefills = frozen_model_input.attn_metadata.num_prefills

        model_input = StatefulModelInput(
            frozen_model_input=frozen_model_input,
            num_seqs=num_seqs,
            num_queries=num_queries,
            num_single_step_prefills=num_single_step_prefills,
        )

        return model_input

    def make_model_input_from_broadcasted_tensor_dict(self, tensor_dict):
        if tensor_dict.get("input_positions", None) is None:
            return StatefulModelInput(
                frozen_model_input=ModelInputForGPUWithSamplingMetadata(),
                num_seqs=0,
                num_queries=0,
                num_single_step_prefills=0,
            )

        return super().make_model_input_from_broadcasted_tensor_dict(tensor_dict)

    def _advance_step(self, model_input, out):
        frozen_model_input = model_input.frozen_model_input
        if frozen_model_input.input_positions is None:
            return model_input

        return super()._advance_step(model_input, out)

    def _final_process_outputs(self, model_input, output_proc_callback):
        frozen_model_input = model_input.frozen_model_input
        if frozen_model_input.input_positions is None:
            outputs = []
            for output in model_input.cached_outputs:
                output.sampler_output.outputs = [
                    CompletionSequenceGroupOutput(
                        samples=[
                            SequenceOutput(
                                parent_seq_id=list(seq_group_metadata.seq_data.keys())[
                                    0
                                ],
                                output_token=0,
                                logprobs={0: Logprob(0.0)},
                            )
                        ],
                        prompt_logprobs=None,
                    )
                    for seq_group_metadata in self._seq_group_metadata_list
                ]
                outputs.append(output.sampler_output)
            return outputs

        return super()._final_process_outputs(model_input, output_proc_callback)


class GCUMultiStepWorker(GCUWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_model_runner = self.model_runner
        # for multi-step model, wrap the model runner with MultiStepModelRunner
        self.model_runner = GCUMultiStepModelRunner(
            base_model_runner,
            vllm_config=base_model_runner.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=base_model_runner.is_driver_worker,
        )

        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[Optional[MultiStepState]] = [
            None
        ] * pipeline_parallel_size
        self.temp_output = None

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
        """
        Get the driver input and broadcast it to other workers.
        """
        assert self.is_driver_worker
        virtual_engine = execute_model_req.virtual_engine
        is_first_multi_step = execute_model_req.is_first_multi_step
        if is_first_multi_step:
            # on first step we prepare the worker input and model input normally
            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req
            )
            model_input: StatefulModelInput = self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.virtual_engine,
                execute_model_req.finished_requests_ids,
            )

            if execute_model_req.async_callback:
                model_input.frozen_model_input = dataclasses.replace(  # type: ignore
                    model_input.frozen_model_input,
                    async_callback=execute_model_req.async_callback,
                )
        else:
            # on subsequent steps we reuse the worker input and model input
            multi_step_state = self.multi_step_states[virtual_engine]
            worker_input = multi_step_state.worker_input
            model_input = multi_step_state.model_input
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None
            if frozen_model_input.attn_metadata is not None:
                # clear the cached metadata so that it can be recomputed on
                # the workers.
                frozen_model_input.attn_metadata._cached_prefill_metadata = None
                frozen_model_input.attn_metadata._cached_decode_metadata = None

        model_input.is_first_multi_step = is_first_multi_step
        model_input.is_last_step = execute_model_req.is_last_step

        if not is_first_multi_step:
            # we broadcast the last sampled token ids to all TP workers so they
            # can update their model input metadata in-place.
            self._prepare_last_sampled_token_ids_for_tp_workers(
                execute_model_req=execute_model_req, model_input=model_input
            )

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_tensor_dict(broadcast_data, src=0)

        # Retuning empty dict here to keep this compatible with
        # `LocalOrDistributedWorkerBase._get_driver_input_and_broadcast`
        return model_input, worker_input, {}

    def _prepare_last_sampled_token_ids_for_tp_workers(
        self,
        execute_model_req: ExecuteModelRequest,
        model_input: StatefulModelInput,
    ) -> None:
        """
        Prepare the last sampled token ids for TP workers. If it's the last
        PP rank, then the last sampled token ids are already in the model_input.
        If it is NOT the last PP rank, then we need to get the last sampled
        token that is cached in the execute_model_req.
        """
        if get_pp_group().is_last_rank:
            assert (
                model_input.cached_outputs[-1].sampler_output.sampled_token_ids is None
            )
            model_input.last_sampled_token_ids = model_input.cached_outputs[
                -1
            ].sampled_token_ids
            # free sampled token ids from the previous step if it has been
            # pythonized. Cannot free the last sampled token ids because
            # we need it for GPU advance_step.
            for output in model_input.cached_outputs[:-1]:
                if output.pythonized:
                    output.sampled_token_ids = None
        else:
            # otherwise we need to get the cached sampled token ids from the
            # execute_model_req
            assert execute_model_req.last_sampled_token_ids is not None
            model_input.last_sampled_token_ids = (
                execute_model_req.last_sampled_token_ids.cuda()
            )
            model_input.add_sampler_output(
                SamplerOutput(outputs=[], sampled_token_ids=None),
                model_input.last_sampled_token_ids,
            )

            # free sampled token ids from the previous step.
            # TODO(will) we could reuse the sampled token ids tensor from
            # the previous step instead.
            for output in model_input.cached_outputs[:-1]:
                output.sampled_token_ids = None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[Tuple[StatefulModelInput, WorkerInput, Dict[str, torch.Tensor]]]:
        """
        Depending on the current state of the request and multi step worker,
        this method may skip the normal _prepare_model_input and
        _prepare_worker_input methods and instead used cached values.
        """
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None

            virtual_engine = execute_model_req.virtual_engine
            (model_input, worker_input, kwargs) = self._get_driver_input_and_broadcast(
                execute_model_req
            )
            assert isinstance(model_input, StatefulModelInput)
            if execute_model_req.is_first_multi_step:
                # cache the worker input and model input for the next steps
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input=worker_input, model_input=model_input
                )
        # if TP workers
        else:
            broadcast_data = self._get_worker_input_from_broadcast()
            # if the driver has sent an empty input, we should stop the worker
            # loop
            if broadcast_data is None:
                return None
            model_input, worker_input, kwargs = broadcast_data
            assert isinstance(model_input, StatefulModelInput)
            virtual_engine = worker_input.virtual_engine
            if model_input.is_first_multi_step:
                pass
                # TODO(will) Can cache the worker input and model input for the
                # next steps. See below for details
            else:
                # TODO(will) possible to also cache and reuse the cached worker
                # input and model input. The idea is essentially the delta
                # optimization for model_inputs. Where the TP workers can cache
                # the model input states and we only broadcast the delta need
                # for the next step (sampled_token_ids from the previous step)

                assert isinstance(model_input, StatefulModelInput)
                # we need to update the last sampled token ids in the model
                # input for the workers so that they can run inplace
                # advance_step
                model_input.add_sampler_output(
                    SamplerOutput(outputs=[], sampled_token_ids=None),
                    model_input.last_sampled_token_ids,
                )

        assert model_input is not None
        assert worker_input is not None
        return model_input, worker_input, kwargs
