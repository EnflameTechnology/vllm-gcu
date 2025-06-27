import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
import torch_gcu

from vllm.logger import init_logger
from vllm.distributed.communication_op import (broadcast_tensor_dict, tensor_model_parallel_gather, get_tp_group)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.sequence import (VLLM_INVALID_TOKEN_ID,
                           CompletionSequenceGroupOutput, ExecuteModelRequest,
                           HiddenStates, SequenceOutput,
                           Logprob)
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler, SpecDecodeStochasticBaseSampler)
from vllm.spec_decode.interfaces import SpeculativeScorer
from vllm.spec_decode.mqa_scorer import MQAScorer
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler)
from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker, prepare_prefill_hidden_states
from vllm.worker.worker_base import WorkerBase
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.mlp_speculator_worker import MLPSpeculatorWorker
from vllm.spec_decode.medusa_worker import MedusaWorker
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.util import nvtx_range
from vllm.utils import resolve_obj_by_qualname

from .smaller_tp_proposer_worker import SmallerTpProposerGCUWorker
from .utils import patch_data_parallel_group
from .metrics import AsyncMetricsGCUCollector

logger = init_logger(__name__)

def create_spec_worker(*args, **kwargs) -> "SpecDecodeGCUWorker":
    """Helper method that is the entrypoint for Executors which use
    WorkerWrapper. It constructs a SpecDecodeWorker from the speculative config.
    """
    vllm_config: VllmConfig = kwargs.get("vllm_config")
    speculative_config: SpeculativeConfig = vllm_config.speculative_config
    assert speculative_config is not None

    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError("Speculative decoding is currently "
                                  "incompatible with pipeline parallelism")

    draft_worker_kwargs = kwargs.copy()

    kwargs["model_runner_cls"] = TargetModelRunner
    target_worker_config = copy.deepcopy(vllm_config)
    target_worker_config.parallel_config.worker_cls =\
        target_worker_config.parallel_config.sd_worker_cls
    cls = resolve_obj_by_qualname(
        target_worker_config.parallel_config.worker_cls)
    target_worker = cls(*args, **kwargs)
    # Set the disable_logprobs variable in the TargetModelRunner instance
    # as per its value specified in the SpeculativeConfig.
    target_worker.model_runner.disable_logprobs =\
         speculative_config.disable_logprobs

    draft_worker_config = copy.deepcopy(vllm_config)
    draft_worker_config.model_config = speculative_config.draft_model_config
    draft_worker_config.quant_config = VllmConfig._get_quantization_config(
        draft_worker_config.model_config,
        vllm_config.load_config,
    )
    speculative_config.draft_parallel_config.worker_cls =\
        draft_worker_config.parallel_config.sd_worker_cls
    draft_worker_config.parallel_config = speculative_config.draft_parallel_config  # noqa
    # Only support none ep for now, get rid of communication between ep ranks. Otherwise in dp will get dead-locked.
    draft_worker_config.parallel_config.data_parallel_size = 1
    draft_worker_config.parallel_config.data_parallel_rank = 0
    draft_worker_config.parallel_config.enable_expert_parallel = False
    draft_worker_config.parallel_config.disable_custom_all_reduce = True
    # TODO allow draft-model specific load config.

    # Override draft-model specific worker args.
    draft_worker_kwargs.update(
        vllm_config=draft_worker_config,
        ngram_prompt_lookup_max=speculative_config.ngram_prompt_lookup_max,
        ngram_prompt_lookup_min=speculative_config.ngram_prompt_lookup_min,
    )

    spec_decode_worker = SpecDecodeGCUWorker.create_worker(
        scorer_worker=target_worker,
        draft_worker_kwargs=draft_worker_kwargs,
        disable_mqa_scorer=speculative_config.speculative_disable_mqa_scorer,
        disable_by_batch_size=speculative_config.
        speculative_disable_by_batch_size,
        draft_token_acceptance_method=speculative_config.
        draft_token_acceptance_method,
        typical_acceptance_sampler_posterior_threshold=speculative_config.
        typical_acceptance_sampler_posterior_threshold,
        typical_acceptance_sampler_posterior_alpha=speculative_config.
        typical_acceptance_sampler_posterior_alpha,
        disable_logprobs=speculative_config.disable_logprobs,
        disable_log_stats=speculative_config.disable_log_stats,
        num_speculative_tokens=speculative_config.num_speculative_tokens,
    )

    return spec_decode_worker

class SpecDecodeGCUWorker(SpecDecodeWorker):
    @classmethod
    def create_worker(
        cls,
        scorer_worker: WorkerBase,
        draft_worker_kwargs: Dict[str, Any],
        disable_mqa_scorer: bool,
        disable_by_batch_size: Optional[int],
        draft_token_acceptance_method: str,
        typical_acceptance_sampler_posterior_threshold: float,
        typical_acceptance_sampler_posterior_alpha: float,
        disable_logprobs: bool,
        disable_log_stats: bool,
        num_speculative_tokens: int,
    ) -> "SpecDecodeGCUWorker":

        allow_zero_draft_token_step = True
        enable_lm_head_weight_load = False
        num_spec_prefill_steps = 1
        ngram_prompt_lookup_max = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_max"))
        ngram_prompt_lookup_min = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_min"))
        draft_model_config = draft_worker_kwargs["vllm_config"].model_config
        draft_parallel_config: ParallelConfig = draft_worker_kwargs[
            'vllm_config'].parallel_config
        if ngram_prompt_lookup_max > 0:
            draft_worker_kwargs[
                "device_type"] = scorer_worker.device_config.device.type
            proposer_worker = NGramWorker(**draft_worker_kwargs)
            proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                                  ngram_prompt_lookup_max)
        else:
            draft_tp = draft_parallel_config.tensor_parallel_size
            target_tp = scorer_worker.parallel_config.tensor_parallel_size

            if draft_model_config.hf_config.model_type == "mlp_speculator":
                proposer_worker = MLPSpeculatorWorker(**draft_worker_kwargs)
            elif draft_model_config.hf_config.model_type == "medusa":
                proposer_worker = MedusaWorker(**draft_worker_kwargs)
            else:
                if draft_tp == 1:
                    # if current_platform.is_cuda_alike():
                    #     draft_worker_kwargs[
                    #         "model_runner_cls"] = TP1DraftModelRunner
                    pass
                else:
                    if draft_model_config.hf_config.model_type == "eagle":
                        raise NotImplementedError(
                            f"{draft_model_config.hf_config.model_type} "
                            "does not support TP > 1 yet")

                    allow_zero_draft_token_step = False

                # Load lm_head weight for eagle in init_device
                if draft_model_config.hf_config.model_type == "eagle":
                    enable_lm_head_weight_load = True

                proposer_worker = MultiStepWorker(**draft_worker_kwargs)
                if draft_model_config.hf_config.model_type == "deepseek_mtp":
                    num_spec_prefill_steps = \
                        draft_model_config.hf_config.n_predict

            proposer_worker = SmallerTpProposerGCUWorker.maybe_wrap_worker(
                proposer_worker, draft_tp, target_tp)

        logger.info("Configuring SpecDecodeWorker with proposer=%s",
                    type(proposer_worker))

        spec_decode_sampler: SpecDecodeBaseSampler = None
        if draft_token_acceptance_method == "rejection_sampler":
            spec_decode_sampler = RejectionSampler()
        elif draft_token_acceptance_method == "typical_acceptance_sampler":
            spec_decode_sampler = TypicalAcceptanceSampler(
                posterior_threshold=\
                    typical_acceptance_sampler_posterior_threshold,
                posterior_alpha=typical_acceptance_sampler_posterior_alpha,
            )
        logger.info(
            "[Speculative Decoding] Configuring"
            " SpecDecodeWorker with sampler=%s", type(spec_decode_sampler))

        if not disable_mqa_scorer:
            if scorer_worker.model_runner.attn_backend.get_name(
            ) != "FLASH_ATTN":
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "MQA is only available with flash attn backend.")

            if draft_model_config and \
                draft_model_config.max_model_len < \
                    scorer_worker.model_config.max_model_len:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "draft model max_model_len is smaller than the target "
                    "model max_model_len.")

            if not scorer_worker.model_runner.model_config.enforce_eager:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "target model is not running in eager mode.")

        return SpecDecodeGCUWorker(
            proposer_worker,
            scorer_worker,
            disable_mqa_scorer=disable_mqa_scorer,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            disable_by_batch_size=disable_by_batch_size,
            spec_decode_sampler=spec_decode_sampler,
            allow_zero_draft_token_step=allow_zero_draft_token_step,
            enable_lm_head_weight_load=enable_lm_head_weight_load,
            num_spec_prefill_steps=num_spec_prefill_steps)

    def __init__(
        self,
        proposer_worker: ProposerWorkerBase,
        scorer_worker: WorkerBase,
        spec_decode_sampler: SpecDecodeBaseSampler,
        disable_mqa_scorer: bool = False,
        disable_logprobs: bool = False,
        disable_log_stats: bool = False,
        metrics_collector: Optional[AsyncMetricsGCUCollector] = None,
        disable_by_batch_size: Optional[int] = None,
        allow_zero_draft_token_step: Optional[bool] = True,
        enable_lm_head_weight_load: Optional[bool] = False,
        num_spec_prefill_steps: int = 1,
    ):
        metrics_collector = AsyncMetricsGCUCollector(
            spec_decode_sampler
        ) if metrics_collector is None else metrics_collector
        super().__init__(
            proposer_worker=proposer_worker,
            scorer_worker=scorer_worker,
            spec_decode_sampler=spec_decode_sampler,
            disable_mqa_scorer=disable_mqa_scorer,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            metrics_collector=metrics_collector,
            disable_by_batch_size=disable_by_batch_size,
            allow_zero_draft_token_step=allow_zero_draft_token_step,
            enable_lm_head_weight_load=enable_lm_head_weight_load,
            num_spec_prefill_steps=num_spec_prefill_steps
        )

    def init_device(self) -> None:
        """Initialize both scorer and proposer models.
        """
        # The scorer worker model is initialized first in case the proposer
        # model has a smaller TP degree than the target worker.
        self.scorer_worker.init_device()
        with patch_data_parallel_group():
            self.proposer_worker.init_device()

        # NOTE(cade): load_model is not part of the WorkerBase interface.
        self.scorer_worker.load_model()
        self.proposer_worker.load_model()

        if self._enable_lm_head_weight_load:
            # NOTE(Shangming): gather lm_head weight when tp enabled
            target_lm_head_weight: torch.Tensor = tensor_model_parallel_gather(
                self.scorer_worker.model_runner.model_runner.model.lm_head.\
                    weight.data,
                    dim=0,
            )

            self.proposer_worker.maybe_load_lm_head_weight(
                target_lm_head_weight)

        self._metrics.init_tensors(self.rank, device_type=self.device)
        if model_parallel_is_initialized():
            self.spec_decode_sampler.init_tensors(get_tp_group().local_rank,
                                                  device_type=self.device)
        else:
            self.spec_decode_sampler.init_tensors(self.rank,
                                                  device_type=self.device)

        scorer_cls: Type[SpeculativeScorer]
        if self.disable_mqa_scorer:
            scorer_cls = BatchExpansionTop1Scorer
            logger.info("[Speculative Decoding] Use batch "
                        "expansion for scoring proposals.")
        else:
            scorer_cls = MQAScorer
            logger.info(
                "[Speculative Decoding] Use MQA scorer for scoring proposals.")

        self.scorer = scorer_cls(scorer_worker=self.scorer_worker,
                                 device=self.device,
                                 vocab_size=self._vocab_size)

        self._configure_model_sampler_for_spec_decode()

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform speculative decoding on the input batch.
        """
        if self.rank != self._driver_rank:
            self._run_non_driver_rank()
            return []

        if execute_model_req is None:
            # This signals that there's no more requests to process for now.
            # All workers are running infinite loop with broadcast_tensor_dict,
            # and it stops the loop when the driver broadcasts an empty input.
            # Send an empty input to notify all other workers to stop their
            # execution loop.
            broadcast_tensor_dict({}, src=0)
            return []

        self._track_finished_requests(execute_model_req)
        disable_all_speculation = self._should_disable_all_speculation(
            execute_model_req)
        num_lookahead_slots = execute_model_req.num_lookahead_slots
        all_prompt = True
        atleast_one_prompt = False
        all_zero_spec_tokens = True
        for sgm in execute_model_req.seq_group_metadata_list:
            all_prompt = all_prompt and sgm.is_prompt
            atleast_one_prompt = atleast_one_prompt or sgm.is_prompt
            all_zero_spec_tokens = all_zero_spec_tokens and (
                sgm.num_speculative_tokens == 0)

        if all_prompt and execute_model_req.seq_group_metadata_list:
            assert num_lookahead_slots == 0, (
                "Prompt only runs should have num_lookahead_slots equal to 0. "
                "This should never happen, please file a bug at "
                "https://github.com/vllm-project/vllm/issues")
        # Speculative decoding is disabled in the following cases:
        # 1. Prefill phase: Speculative decoding is not
        #    used during the prefill phase.
        # 2. Auto-disable enabled: The running queue size exceeds
        #    the specified threshold.
        # 3. No request: There are no requests in the batch, or
        #    none of the requests in the batch have spec decoding enabled.
        # In any of these cases, the proposer and scorer workers
        # are called normally.
        # We expect `num_speculative_tokens` to be None for prefills.
        no_spec = (num_lookahead_slots == 0 or disable_all_speculation
                   or all_zero_spec_tokens)

        all_idle = True
        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            if not (
                seq_group_metadata.sampling_params.extra_args
                and seq_group_metadata.sampling_params.extra_args.get("is_idle", None)
            ):
                all_idle = False


        # Broadcast how many lookahead slots are scheduled for this step, and
        # whether all speculation is disabled, to all non-driver workers.

        # This is required as if the number of draft model runs changes
        # dynamically, the non-driver workers won't know unless we perform a
        # communication to inform them.

        # no_spec is used to signal non-driver worker about prefill vs decode
        # stage. This is needed to ensure that order of execution of proposer
        # and scorer is same in both driver and non-driver workers (i.e.,
        # scorer -> proposer for prefill and proposer -> scorer in decode). This
        # order is needed to support models like EAGLE that take scorer states
        # as inputs.
        broadcast_dict = dict(
            num_lookahead_slots=num_lookahead_slots,
            no_spec=no_spec,
            disable_all_speculation=disable_all_speculation,
            all_idle=all_idle,
            # When both chunked prefill and speculative decoding are enabled
            # it is possible that the same batch contains both prefill
            # and decodes. If that happens in the scorer we run the batch
            # as one single forward pass. However, in the proposer we
            # run them as 2 different batches - one for prefill and
            # the other for decodes. The variable indicates to the non-driver
            # worker that there are prefills as part of the speculative batch
            # and hence it needs to run an extra prefill forward pass.
            run_spec_proposer_for_prefill=atleast_one_prompt,
        )
        broadcast_tensor_dict(broadcast_dict, src=self._driver_rank)

        assert execute_model_req.seq_group_metadata_list is not None, (
            "speculative decoding requires non-None seq_group_metadata_list")

        self._maybe_disable_speculative_tokens(
            disable_all_speculation, execute_model_req.seq_group_metadata_list)

        if no_spec:
            return self._run_no_spec(execute_model_req,
                                     skip_proposer=disable_all_speculation)
        return self._run_speculative_decoding_step(execute_model_req,
                                                   num_lookahead_slots)

    def _run_non_driver_rank(self) -> bool:
        """Run proposer and verifier model in non-driver workers. This is used
        for both speculation cases (num_lookahead_slots>0) and non-speculation
        cases (e.g. prefill).

        Returns True if there are remaining sequences to process.
        """
        assert self.rank != self._driver_rank

        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:
            return False
        num_lookahead_slots = data["num_lookahead_slots"]

        # In case of prefill, scorer_worker has to be run before proposer so
        # that the hidden states can be propagated to proposer when needed.
        if data["no_spec"]:
            self.scorer_worker.execute_model()

        if not data["disable_all_speculation"]:
            # Even if num_lookahead_slots is zero, we want to run the
            # proposer model as it may have KV.
            #
            # We run the proposer once per lookahead slot. In the future we
            # should delegate how many times it runs to the proposer.
            if data["no_spec"] and data["all_idle"]:
                return True

            for _ in range(max(num_lookahead_slots, 1)):
                self.proposer_worker.execute_model()

        if not data["no_spec"]:
            self.scorer_worker.execute_model()
            if data["run_spec_proposer_for_prefill"]:
                self.proposer_worker.execute_model()

        return True

    @nvtx_range("spec_decode_worker._run_no_spec")
    def _run_no_spec(self, execute_model_req: ExecuteModelRequest,
                     skip_proposer: bool) -> List[SamplerOutput]:
        """Run a single generation step without any speculation. The input is
        sent to the proposer and scorer model so that the KV cache is consistent
        between the two. When skip_proposer is True, the proposer model is
        not called, meaning that the kv-cache in proposer for requests is not
        updated, so they cannot enable spec decode in the rest decoding.
        """
        sampler_output = self.scorer_worker.execute_model(execute_model_req)
        assert len(sampler_output) == 1
        sampler_output = sampler_output[0]
        # Store hidden states from target model execution, BxD.
        hidden_states = sampler_output.hidden_states
        # for idle
        if hidden_states is not None and len(hidden_states) == 0:
            seq_data_entries = [
                (seq_id, seq_data) for sg in \
                execute_model_req.seq_group_metadata_list \
                for seq_id, seq_data in sg.seq_data.items()
            ]
            output = [
                SamplerOutput(
                    outputs=[
                            CompletionSequenceGroupOutput(
                                    samples=[
                                        SequenceOutput(
                                            seq_data_entries[idx][0], 0,
                                            {0: Logprob(logprob=float('inf'), rank=None, decoded_token=None)})
                                    ], prompt_logprobs=None)
                        for idx, sgm in enumerate(execute_model_req.seq_group_metadata_list)
                    ]
                )
            ]

            return output
        if hidden_states is not None:
            # Only decodes and prefill terminal chunks need a hidden state.
            seq_group_meta_with_hidden = [
                sg for sg in execute_model_req.seq_group_metadata_list
                if sg.do_sample
            ]
            if any(seq.is_prompt for seq in seq_group_meta_with_hidden):
                # Drop hidden_states with no prediction (eg non-terminal chunks)
                hidden_states = hidden_states[
                    torch.where(sampler_output.sampled_token_ids -
                                VLLM_INVALID_TOKEN_ID)[0]]
            if self.previous_hidden_states is None and len(
                    seq_group_meta_with_hidden):
                self.previous_hidden_states = HiddenStates(
                    hidden_states, seq_group_meta_with_hidden)
            elif self.previous_hidden_states and len(
                    seq_group_meta_with_hidden):
                self.previous_hidden_states.update(hidden_states,
                                                   seq_group_meta_with_hidden)

        if not skip_proposer:
            # We prepare the prefill hidden states here so that there no
            # additional complexity in worker for spec_decode vs non_spec_decode
            # flow and execute_model doesn't need additional modifications.
            execute_model_req.previous_hidden_states = \
                prepare_prefill_hidden_states(
                    sampler_output.prefill_hidden_states)
            for i in range(self._num_spec_prefill_steps):
                execute_model_req.spec_step_idx = i
                self.proposer_worker.execute_model(execute_model_req)

        sampler_output_to_return = (self._serialize_sampler_output_no_logprobs(
            execute_model_req=execute_model_req, sampler_output=sampler_output)
                                    if self._disable_logprobs else
                                    [sampler_output])

        # Clear device tensors from sampler output. This reduces communication
        # overhead when the engine runs in a different process than the workers.
        sampler_output.sampled_token_probs = None
        sampler_output.sampled_token_ids = None
        sampler_output.logprobs = None
        return sampler_output_to_return
