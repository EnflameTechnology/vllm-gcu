
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from typing import List, Optional, Tuple, Union, Callable, Optional, Tuple
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase
)
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.utils import set_weight_attrs
from abc import abstractmethod

@torch.jit.script
def cal_scale(amax, fp_max, scale):
    margin = 0
    exp = torch.floor(torch.log2(fp_max / amax)) - margin
    sf = torch.round(torch.pow(2, torch.abs(exp)))
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    scale = torch.where(exp < 0, 1 / sf, sf)
    scale_inv = torch.reciprocal(scale)
    return scale, scale_inv


instances = {}


def singleton(cls):
    global instances

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def reset_singleton():
    global instances
    instances = {}

@singleton
class QuantFp8:

    def __init__(self, device):
        self.fp_max = torch.tensor([448.0], device=device)
        self.device = device
        self.scale = torch.tensor([1.0], device=self.device)
        pass

    @staticmethod
    def quantize_v1(weight, bits):
        if bits == 8:
            amax = weight.abs().max()
            fp_max = torch.tensor([448.0]).to(weight.device)
            margin = 0
            scale = torch.tensor([1.0]).to(weight.device)

            exp = torch.floor(torch.log2(fp_max / amax)) - margin
            sf = torch.round(torch.pow(2, torch.abs(exp)))
            sf = torch.where(amax > 0.0, sf, scale)
            sf = torch.where(torch.isfinite(amax), sf, scale)
            scale = torch.where(exp < 0, 1 / sf, sf)

            qweight = (weight.to(torch.float32) * scale).to(
                torch.float8_e4m3fn)
            scale = torch.reciprocal(scale)
            # print(f"amax={amax},scalse={scale}")
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
        return qweight, scale

    def quantize(self, weight, bits, weight_scale, use_offline_input_scales):
        if bits == 8:
            amax = torch.empty(1, dtype=torch.float32, device=self.device)
            scale = torch.tensor([1.0], device=self.device)
            torch.ops.OptimusFp8.abs_max_nan_to_inf(weight, amax)
            if weight_scale is None or not use_offline_input_scales:
                scale, scale_inv = cal_scale(amax, self.fp_max, scale)
            else:
                scale, scale_inv = weight_scale, torch.reciprocal(weight_scale)

            qweight = torch.ops.OptimusFp8.quantize(weight, scale, None,
                                                    torch.float8_e4m3fn)
            # print(f"scale={scale},self.amax={self.amax}")
            return qweight, scale_inv
        else:
            raise ValueError(f"Unsupported bit width: {bits}")

    def get_quant_scale(self, tensor):
        amax = torch.empty(1, dtype=torch.float32, device=tensor.device)
        torch.ops.OptimusFp8.abs_max_nan_to_inf(tensor, amax)
        scale, _ = cal_scale(amax, self.fp_max, self.scale)
        return scale


def dynamic_fp8_pertensor_quantize(tensor):
    # amax = torch.empty(1, dtype=torch.float32, device=tensor.device)
    # scale = torch.tensor([1.0], device=tensor.device)
    # fp_max = torch.tensor([448.0], device=tensor.device)
    # torch.ops.OptimusFp8.abs_max_nan_to_inf(tensor, amax)
    # scale, _ = cal_scale(amax, fp_max, scale)
    # return scale
    quant = QuantFp8(tensor.device)
    return quant.get_quant_scale(tensor)

class OptimusRMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self,
                x: torch.Tensor,
                residual: Optional[torch.Tensor] = None,
                output: Optional[torch.Tensor] = None,
                fp16_out: bool = False) -> torch.Tensor:
        if residual is not None:
            assert output is None
            from vllm import _custom_ops as ops

            assert not fp16_out
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        else:
            if fp16_out:
                if output is None:
                    output = torch.empty_like(x).half()
                else:
                    output = output.half()
            return torch.ops.Optimus.rms_norm(x,
                                              self.weight,
                                              self.variance_epsilon,
                                              out=output)

class OptimusSiluAndMul(nn.Module):

    def forward(self,
                x: torch.Tensor,
                output: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.ops.Optimus.SiluDot_forward(x, out=output)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    return torch.nn.functional.linear


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              residual: Optional[torch.Tensor] = None,
              output: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        if residual is not None:
            assert output is None or output is residual
            if get_tensor_model_parallel_world_size(
            ) > 1 and get_tensor_model_parallel_rank() != 0:
                beta = 0.0
            else:
                beta = 1.0
            # optimize cuda memory usage
            if x.dim() == 2:
                torch.addmm(residual, x, weight.t(), beta=beta, out=residual)
            elif x.dim() >= 3:
                hx = x.size(-1)
                hr = residual.size(-1)
                torch.addmm(residual.view(-1, hr),
                            x.view(-1, hx),
                            weight.t(),
                            beta=beta,
                            out=residual.view(-1, hr))
            else:
                raise AssertionError(
                    "unrecognized tensor dimensions: {}".format(x.dim()))
            if bias is not None:
                residual += bias
            return residual
        else:
            if output is not None:
                if bias is not None:  # always separate bias add when output is provided
                    torch.matmul(x, weight.t(), out=output)
                    output.add_(bias)
                    return output
                return torch.matmul(x, weight.t(), out=output)
            else:
                return dispatch_unquantized_gemm()(x, layer.weight, bias)



class UnquantizedMoELinearMethod(LinearMethodBase):
    """MoE Linear method without quantization.
    """

    def __init__(self):
        self.quant_config = None

    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int],
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       num_experts: Optional[int] = None,
                       **extra_weight_attrs):
        weight = Parameter(torch.empty(num_experts,
                                       sum(output_partition_sizes),
                                       input_size_per_partition,
                                       device=torch.cuda.current_device(),
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 2, "output_dim": 1})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the weights to the input tensor."""
        raise NotImplementedError

class MergedColumnParallelMoELinear(MergedColumnParallelLinear):

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_sizes: List[int],
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        torch.nn.Module.__init__(self)
        output_size = sum(output_sizes)
        self.num_experts = num_experts
        self.output_sizes = output_sizes
        self.input_size = input_size
        self.output_size = sum(output_sizes)
        tp_size = get_tensor_model_parallel_world_size()
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [
            divide(output_size, tp_size) for output_size in self.output_sizes
        ]
        self.gather_output = False
        if output_sizes is None:
            output_sizes = [output_size]
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if quant_config is None:
            self.quant_method = UnquantizedMoELinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(
                self, prefix=prefix
            )
            # FIXME(ys): hack for moe
            if isinstance(self.quant_method, UnquantizedLinearMethod):
                self.quant_method = UnquantizedMoELinearMethod()

        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            self.params_dtype,
            self.num_experts,
            weight_loader=self.weight_loader,
        )
        self.register_parameter("bias", None)

    def forward(
        self,
        input_,
        output: Optional[torch.Tensor] = None,
        expert_idx: int = -1,
    ):
        if isinstance(self.quant_method, UnquantizedMoELinearMethod):
            # use optimus moe_ffn outside
            return
        bias = None
        assert self.quant_method is not None

        output = self.quant_method.apply(
            self, input_, bias, expert_idx=expert_idx, output=output
        )
        return output


class RowParallelMoELinear(RowParallelLinear):

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        torch.nn.Module.__init__(self)
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.reduce_results = False
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedMoELinearMethod()
            )
        else:
            self.quant_method = quant_config.get_quant_method(
                self, prefix=prefix
            )
            # FIXME(ys): hack for moe
            if isinstance(self.quant_method, UnquantizedLinearMethod):
                self.quant_method = UnquantizedMoELinearMethod()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size_per_partition,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            self.num_experts,
            weight_loader=self.weight_loader,
        )
        self.register_parameter("bias", None)

    def forward(  # type: ignore[override]
        self,
        input_,
        residual=None,
        expert_idx: int = -1,
        output: Optional[torch.Tensor] = None,
    ):
        if isinstance(self.quant_method, UnquantizedMoELinearMethod):
            # use optimus moe_ffn outside
            return
        bias = None
        assert self.quant_method is not None
        output = self.quant_method.apply(
            self, input_, bias, expert_idx=expert_idx, output=output
        )
        return output