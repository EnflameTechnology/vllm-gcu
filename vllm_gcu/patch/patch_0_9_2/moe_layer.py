import torch
from unittest.mock import patch
from typing import Optional

from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase, FusedMoEConfig, FusedMoEPrepareAndFinalize,
    FusedMoEPermuteExpertsUnpermute)
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoEOri
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.distributed import get_ep_group
from vllm.logger import init_logger

from vllm_gcu.kernels.prepare_finalize import AlltoAllSelector, MoEPrepareAndFinalizeNoEP

from vllm_gcu.kernels.modular_kernel import FusedMoEModularKernel
from vllm_gcu.kernels.modular_experts import TritonExpertsPad

logger = init_logger(__name__)


def init_prepare_finalize(self, moe: FusedMoEConfig,
                          quant_config: Optional[QuantizationConfig]):
    if moe.use_pplx_kernels or moe.use_deepep_ht_kernels or moe.use_deepep_ll_kernels:
        return FusedMoEMethodBase.init_prepare_finalize(
            self, moe, quant_config)

    # We allow no ep use modular kernel.
    # all2all_manager = get_ep_group().device_communicator.all2all_manager
    # assert all2all_manager is not None

    self.moe = moe

    prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None
    if moe.moe_parallel_config.ep_size > 1:
        prepare_finalize = AlltoAllSelector(None,
                                            moe.moe_parallel_config.dp_size)
    else:
        prepare_finalize = MoEPrepareAndFinalizeNoEP()

    self.topk_indices_dtype = None
    if prepare_finalize is not None:
        logger.debug("%s", prepare_finalize.__class__.__name__)
        self.topk_indices_dtype = prepare_finalize.topk_indices_dtype()

        experts = self.select_gemm_impl(prepare_finalize, moe)
        self.fused_experts = FusedMoEModularKernel(
            prepare_finalize,
            experts,
        )


def select_gemm_impl_unquant(
    self,
    prepare_finalize: FusedMoEPrepareAndFinalize,
    moe: FusedMoEConfig,
) -> FusedMoEPermuteExpertsUnpermute:
    return TritonExpertsPad()


class FusedMoE(FusedMoEOri):

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        shared_experts = getattr(self, "shared_experts", None)
        routed_scaling_factor = getattr(self, "routed_scaling_factor", 1.0)

        if hasattr(self.quant_method, 'fused_experts') and \
            isinstance(self.quant_method.fused_experts, FusedMoEModularKernel):
            self.quant_method.fused_experts.prepare_finalize.set_shared_experts(
                shared_experts, routed_scaling_factor)
        return super().forward_impl(hidden_states, router_logits)


# yapf: disable
patch("vllm.model_executor.layers.fused_moe.layer.FusedMoE", FusedMoE).start()
patch("vllm.model_executor.layers.fused_moe.FusedMoE", FusedMoE).start()
patch("vllm.model_executor.layers.fused_moe.layer.FusedMoEMethodBase.init_prepare_finalize", init_prepare_finalize).start()
patch("vllm.model_executor.layers.fused_moe.FusedMoEMethodBase.init_prepare_finalize", init_prepare_finalize).start()
patch("vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.select_gemm_impl", select_gemm_impl_unquant).start()
