#!/usr/bin/env python
# coding=utf-8
import torch
from vllm.utils import direct_register_custom_op
from vllm.platforms import current_platform
import vllm_gcu.kernels._custom_ops as ops


@torch.inference_mode()
def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (torch.Tensor): lora'a weight
        output_tensor (torch.Tensor): output tensor
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch, An index of -1 means no lora should be
            applied.
        batches (int): batch size
        add_inputs (bool, optional):  Defaults to False. adds the final lora
            results to the output.
    """
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)

    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,size,rank)
    assert lora_b_weights.is_contiguous()

    ops.dispatch_bgmv(inputs, lora_b_weights, output_tensor,
                      lora_indices_tensor)


@torch.inference_mode()
def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (torch.Tensor): lora'b weight
        output_tensor (torch.Tensor): output tensor
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch, An index of -1 means no lora should be
            applied.
        slice_offst (int): output_tensor's offst
        slice_size (int): current output_tensor's size
        batches (int): batch size
        add_inputs (bool, optional): Defaults to False.
    """
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)

    assert slice_size == lora_b_weights.size(-2)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,size,rank)

    assert lora_b_weights.is_contiguous()

    ops.dispatch_bgmv_low_level(
        inputs,
        lora_b_weights,
        output_tensor,
        lora_indices_tensor,
        slice_offset,
        slice_size,
    )


@torch.inference_mode()
def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_a_weights (torch.Tensor): lora'a weight
        output_tensor (torch.Tensor): output tensor
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        batches (int): batch size
        scaling (float):  Scaling factor.
    """
    assert inputs.dtype == lora_a_weights.dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    assert lora_a_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_a_weights.size(-1)
    assert inputs.is_contiguous()

    if lora_a_weights.ndim == 4:  # shape:(lora_num,1,rank, size)
        assert lora_a_weights.size(1) == 1
        lora_a_weights = lora_a_weights.squeeze(dim=1)
    else:
        assert lora_a_weights.ndim == 3  # shape:(lora_num,rank, size)
    assert lora_a_weights.is_contiguous()
    assert output_tensor.is_contiguous()

    ops.dispatch_bgmv(inputs, lora_a_weights, output_tensor,
                      lora_indices_tensor, scaling)


def sgmv_expand_impl(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
) -> None:
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,size,rank)

    assert lora_b_weights.is_contiguous()

    lora_indices_tensor = lora_indices_tensor.repeat_interleave(seq_len_tensor)
    ops.dispatch_bgmv(inputs, lora_b_weights, output_tensor,
                      lora_indices_tensor, 1.0)


def sgmv_expand_impl_fake(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
) -> None:
    return


direct_register_custom_op(
    op_name="sgmv_expand_impl",
    op_func=sgmv_expand_impl,
    mutates_args=["output_tensor"],
    fake_impl=sgmv_expand_impl_fake,
    dispatch_key=current_platform.dispatch_key,
)


@torch.inference_mode()
def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (torch.Tensor): lora'a weight
        output_tensor (torch.Tensor): output tensor
        b_seq_start_loc (torch.Tensor): (batch_size,). The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g.,if the sequence length is [4, 6], it is
            [0, 4, 10].
        seq_len_tensor (torch.Tensor): (batch_size,). record the sequence
            length of the sequences  in the batch
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        batches (int): batch size
        max_seq_length (int):  The max sequence lengths of the sequences
            in the batch
        add_inputs (bool, optional):  Defaults to False. adds the final lora
            results to the output.
    """
    return torch.ops.vllm.sgmv_expand_impl(
        inputs,
        lora_b_weights,
        output_tensor,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batches,
        max_seq_length,
        token_nums,
        add_inputs,
    )


def sgmv_expand_slice_impl(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
) -> None:
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert slice_size == lora_b_weights.size(-2)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,size,rank)

    assert lora_b_weights.is_contiguous()

    lora_indices_tensor = lora_indices_tensor.repeat_interleave(seq_len_tensor)
    ops.dispatch_bgmv_low_level(
        inputs,
        lora_b_weights,
        output_tensor,
        lora_indices_tensor,
        slice_offset,
        slice_size,
    )


def sgmv_expand_slice_impl_fake(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
) -> None:
    return


direct_register_custom_op(
    op_name="sgmv_expand_slice_impl",
    op_func=sgmv_expand_slice_impl,
    mutates_args=[],
    fake_impl=sgmv_expand_slice_impl_fake,
    dispatch_key=current_platform.dispatch_key,
)


@torch.inference_mode()
def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
) -> None:
    return torch.ops.vllm.sgmv_expand_slice_impl(
        inputs,
        lora_b_weights,
        output_tensor,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batches,
        max_seq_length,
        token_nums,
        slice_offset,
        slice_size,
        add_inputs,
    )


def sgmv_shrink_impl(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    assert inputs.dtype == lora_a_weights.dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    assert lora_a_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_a_weights.size(-1)
    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert inputs.is_contiguous()

    if lora_a_weights.ndim == 4:  # shape:(lora_num,1,rank, size)
        assert lora_a_weights.size(1) == 1
        lora_a_weights = lora_a_weights.squeeze(dim=1)
    else:
        assert lora_a_weights.ndim == 3  # shape:(lora_num,rank, size)
    assert lora_a_weights.is_contiguous()
    assert output_tensor.is_contiguous()
    lora_indices_tensor = lora_indices_tensor.repeat_interleave(seq_len_tensor)
    ops.dispatch_bgmv(inputs, lora_a_weights, output_tensor,
                      lora_indices_tensor, scaling)


def sgmv_shrink_impl_fake(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    return


direct_register_custom_op(
    op_name="sgmv_shrink_impl",
    op_func=sgmv_shrink_impl,
    mutates_args=[],
    fake_impl=sgmv_shrink_impl_fake,
    dispatch_key=current_platform.dispatch_key,
)


@torch.inference_mode()
def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    return torch.ops.vllm.sgmv_shrink_impl(
        inputs,
        lora_a_weights,
        output_tensor,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        batches,
        max_seq_length,
        token_nums,
        scaling,
    )
