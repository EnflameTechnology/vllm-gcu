import torch
from unittest.mock import patch
from typing import Optional, Callable

from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase, FusedMoEConfig, FusedMoEPrepareAndFinalize,
    FusedMoEPermuteExpertsUnpermute)
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoEOri

from vllm.logger import init_logger

from vllm_gcu.kernels.prepare_finalize import AlltoAllSelector, MoEPrepareAndFinalizeNoEP
from vllm_gcu.kernels.modular_experts import TritonExpertsPad
from vllm_gcu.patch.patch_0_11_0.modular_kernel import FusedMoEModularKernel
from vllm_gcu.kernels._custom_ops import eplb_map_to_physical_and_record

logger = init_logger(__name__)


def maybe_make_prepare_finalize(self) -> Optional[FusedMoEPrepareAndFinalize]:
    if self.moe.use_pplx_kernels or self.moe.use_deepep_ht_kernels or self.moe.use_deepep_ll_kernels:
        return FusedMoEMethodBase.maybe_make_prepare_finalize(self)
    # We allow no ep use modular kernel.
    # all2all_manager = get_ep_group().device_communicator.all2all_manager
    # assert all2all_manager is not None

    prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None

    if self.moe.moe_parallel_config.ep_size > 1:
        prepare_finalize = AlltoAllSelector(
            None, self.moe.moe_parallel_config.dp_size)
    else:
        prepare_finalize = MoEPrepareAndFinalizeNoEP()

    return prepare_finalize


def select_gemm_impl_unquant(
    self,
    prepare_finalize,
    layer,
):
    return TritonExpertsPad(self.moe_quant_config)


origin_forward_impl = FusedMoEOri.forward_impl


def forward_impl(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
    if hasattr(self.quant_method, 'fused_experts') and \
        isinstance(self.quant_method.fused_experts, FusedMoEModularKernel):
        self.quant_method.fused_experts.prepare_finalize.set_shared_experts(
            self.shared_experts, self.routed_scaling_factor)
    return origin_forward_impl(self, hidden_states, router_logits)


# yapf: disable
patch("vllm.model_executor.layers.fused_moe.layer.FusedMoE.forward_impl", forward_impl).start()
patch("vllm.model_executor.layers.fused_moe.FusedMoE.forward_impl", forward_impl).start()
patch("vllm.model_executor.layers.shared_fused_moe.shared_fused_moe.FusedMoE.forward_impl", forward_impl).start()
patch("vllm.model_executor.layers.fused_moe.layer.FusedMoEMethodBase.maybe_make_prepare_finalize", maybe_make_prepare_finalize).start()
patch("vllm.model_executor.layers.fused_moe.FusedMoEMethodBase.maybe_make_prepare_finalize", maybe_make_prepare_finalize).start()
patch("vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod.select_gemm_impl", select_gemm_impl_unquant).start()
patch("vllm.model_executor.layers.fused_moe.layer.eplb_map_to_physical_and_record", eplb_map_to_physical_and_record).start()
