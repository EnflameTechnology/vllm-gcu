import pytest
import torch
import unittest.mock as mock
from types import SimpleNamespace

with mock.patch("deep_ep.Buffer"), mock.patch("deep_ep.EventOverlap"), mock.patch(
    "vllm.model_executor.layers.fused_moe.modular_kernel.ExpertTokensMetadata"
):
    from vllm_gcu.kernels.deepep_ht_prepare_finalize import (
        DeepEPHTPrepareAndFinalizeGCU,
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


class TestDeepEPHTPrepareAndFinalizeGCU:
    def _get_configs(self):
        return FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[1, 128],
            a1_scale=torch.tensor([1.0]),
            w1_scale=torch.tensor([1.0]),
            w2_scale=torch.tensor([1.0]),
        )

    def _get_handler(self):
        return DeepEPHTPrepareAndFinalizeGCU(mock.MagicMock(), 1, 1, 8)

    def test_receiver_id_remapping_comparison(self):
        with mock.patch(
            "vllm_gcu.kernels.deepep_ht_prepare_finalize.moe_kernel_quantize_input"
        ) as m_quant:
            torch.manual_seed(0)
            token_data = torch.randn(4, 128)
            token_scales = torch.ones(4, 1)
            ids = torch.tensor([[0], [-1], [2], [5]])
            weights = torch.ones(4, 1)
            a1_s = torch.tensor([1.0])

            torch.set_default_device("cpu")
            m_quant.return_value = (token_data.cpu(), token_scales.cpu())
            h_cpu, c_cpu = self._get_handler(), self._get_configs()
            out_cpu = h_cpu._receiver(
                mock.MagicMock(),
                True,
                (token_data.cpu(), token_scales.cpu()),
                ids.cpu(),
                16,
                [1] * 16,
                weights.cpu(),
                a1_s.cpu(),
                c_cpu,
            )

            torch.set_default_device("gcu")
            m_quant.return_value = (token_data.gcu(), token_scales.gcu())
            h_gcu, c_gcu = self._get_handler(), self._get_configs()
            out_gcu = h_gcu._receiver(
                mock.MagicMock(),
                True,
                (token_data.gcu(), token_scales.gcu()),
                ids.gcu(),
                16,
                [1] * 16,
                weights.gcu(),
                a1_s.gcu(),
                c_gcu,
            )

            assert torch.allclose(out_cpu[3].cpu(), out_gcu[3].cpu())

    def test_prepare_async_scaling_comparison(self):
        torch.manual_seed(0)
        a1 = torch.ones(2, 128)
        w = torch.full((2, 1), 0.5)
        ids = torch.zeros((2, 1), dtype=torch.long)

        torch.set_default_device("cpu")
        h_cpu = self._get_handler()
        h_cpu._do_dispatch = mock.MagicMock()
        h_cpu.prepare_async(
            a1.cpu(), w.cpu(), ids.cpu(), 16, None, True, self._get_configs()
        )
        res_cpu = h_cpu._do_dispatch.call_args[1]["tokens"]

        torch.set_default_device("gcu")
        h_gcu = self._get_handler()
        h_gcu._do_dispatch = mock.MagicMock()
        h_gcu.prepare_async(
            a1.gcu(), w.gcu(), ids.gcu(), 16, None, True, self._get_configs()
        )
        res_gcu = h_gcu._do_dispatch.call_args[1]["tokens"]

        assert torch.allclose(res_cpu.cpu(), res_gcu.cpu())

    def test_receiver_no_scales_comparison(self):
        with mock.patch(
            "vllm_gcu.kernels.deepep_ht_prepare_finalize.moe_kernel_quantize_input"
        ) as m_quant:
            torch.manual_seed(0)
            token_data = torch.randn(4, 128)
            ids = torch.zeros((4, 1), dtype=torch.long)
            weights = torch.ones(4, 1)
            a1_s = torch.tensor([1.0])

            torch.set_default_device("cpu")
            m_quant.return_value = (token_data.cpu(), torch.ones(4, 1))
            h_cpu, c_cpu = self._get_handler(), self._get_configs()
            out_cpu = h_cpu._receiver(
                mock.MagicMock(),
                False,
                token_data.cpu(),
                ids.cpu(),
                16,
                [1] * 16,
                weights.cpu(),
                a1_s.cpu(),
                c_cpu,
            )

            torch.set_default_device("gcu")
            m_quant.return_value = (token_data.gcu(), torch.ones(4, 1).gcu())
            h_gcu, c_gcu = self._get_handler(), self._get_configs()
            out_gcu = h_gcu._receiver(
                mock.MagicMock(),
                False,
                token_data.gcu(),
                ids.gcu(),
                16,
                [1] * 16,
                weights.gcu(),
                a1_s.gcu(),
                c_gcu,
            )

            assert torch.allclose(out_cpu[0].cpu(), out_gcu[0].cpu())

    def test_prepare_async_reshaping_comparison(self):
        with mock.patch(
            "vllm_gcu.kernels.deepep_ht_prepare_finalize.moe_kernel_quantize_input"
        ) as m_quant, mock.patch(
            "vllm_gcu.envs.VLLM_GCU_DEEPEP_USE_FP8_DISPATCH", True
        ):

            torch.manual_seed(0)
            a1 = torch.randn(4, 128)
            scale_1d = torch.tensor([5.0])

            torch.set_default_device("cpu")
            m_quant.return_value = (a1.cpu(), scale_1d.cpu())
            h_cpu = self._get_handler()
            h_cpu._do_dispatch = mock.MagicMock()
            h_cpu.prepare_async(
                a1.cpu(),
                torch.ones(4, 1),
                torch.zeros((4, 1), dtype=torch.long),
                16,
                None,
                False,
                self._get_configs(),
            )

            torch.set_default_device("gcu")
            m_quant.return_value = (a1.gcu(), scale_1d.gcu())
            h_gcu = self._get_handler()
            h_gcu._do_dispatch = mock.MagicMock()
            h_gcu.prepare_async(
                a1.gcu(),
                torch.ones(4, 1).gcu(),
                torch.zeros((4, 1), dtype=torch.long).gcu(),
                16,
                None,
                False,
                self._get_configs(),
            )

            cpu_scale = h_cpu._do_dispatch.call_args[1]["token_scales"]
            gcu_scale = h_gcu._do_dispatch.call_args[1]["token_scales"]

            assert cpu_scale.shape == (1, 1)
            assert torch.allclose(cpu_scale.cpu(), gcu_scale.cpu())
