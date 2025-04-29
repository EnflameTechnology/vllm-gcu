# SPDX-License-Identifier: Apache-2.0

import torch
import os

import vllm.envs as envs
from vllm.distributed.parallel_state import (get_tp_group,
                                             init_model_parallel_group)
from vllm.logger import init_logger
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker

logger = init_logger(__name__)


class SmallerTpProposerGCUWorker(SmallerTpProposerWorker):
    """Class which allows a speculative draft model to run with smaller tensor
    parallel degree than target model.
    This reduces the communication overhead of small draft models.

    To implement this feature, this class differs behavior based on is_dummy
    flag, where dummy means worker that does not participate draft generation.
    Participating workers use a smaller tp group by patching vLLM's tensor
    parallel group temporarily during forward passes of draft models.
    """

    @classmethod
    def maybe_wrap_worker(cls, worker, draft_tensor_parallel_size: int,
                          target_tensor_parallel_size: int):
        """Wrap the worker in a SmallerTpProposerWorker if necessary.
        """
        if draft_tensor_parallel_size == target_tensor_parallel_size:
            return worker

        # gpu ranks that will generate draft tokens together
        dp_rank = envs.VLLM_DP_RANK
        draft_ranks = list(range(dp_rank * target_tensor_parallel_size, dp_rank * target_tensor_parallel_size + draft_tensor_parallel_size))

        logger.info("Wrapping {%s} in {%s}", type(worker), cls)
        return cls(worker, draft_ranks)

