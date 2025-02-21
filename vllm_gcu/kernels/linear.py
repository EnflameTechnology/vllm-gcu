from typing import Optional

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    adjust_bitsandbytes_4bit_shard,
    adjust_marlin_shard,
    adjust_scalar_to_fused_array,
    LinearBase,
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
)

logger = init_logger(__name__)


class MergedReplicatedLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
        )
        self.output_sizes = output_sizes

    def _load_fused_module_from_checkpoint(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        current_shard_offset = 0
        shard_offsets: list[tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader(param, loaded_weight_shard, shard_id)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ):
        if loaded_shard_id is None:
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id < len(self.output_sizes)

        if isinstance(param, BlockQuantScaleParameter):
            from vllm.model_executor.layers.quantization.fp8 import (
                Fp8LinearMethod,
                Fp8MoEMethod,
            )

            assert self.quant_method is not None
            assert isinstance(self.quant_method, (Fp8LinearMethod, Fp8MoEMethod))
            weight_block_size = self.quant_method.quant_config.weight_block_size
            assert weight_block_size is not None
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (
                sum(self.output_sizes[:loaded_shard_id]) + block_n - 1
            ) // block_n
            shard_size = (self.output_sizes[loaded_shard_id] + block_n - 1) // block_n
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]

        if (
            isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
            and param.packed_dim == param.output_dim
        ):
            shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = param.data
        param_data = param_data.narrow(param.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(param.output_dim, 0, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
