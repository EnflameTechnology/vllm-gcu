#!/usr/bin/env python
# coding=utf-8
import torch


def rearrange_uint4_int32_uint8_gptq(
    method, qweight, qzeros, scales, rearrange_group=128
):
    bits = method.quant_config.weight_bits
    group_size = method.quant_config.group_size
    assert bits in [4], "only 4 bit gptq quant is supported."

    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    scales = scales.reshape(-1, 1, scales.shape[-1])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1),
    ).to(torch.int8)
    weight = torch.bitwise_and(weight, (2**bits) - 1)
    weight = weight.reshape(-1, weight.shape[2])

    if qzeros is not None:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
            wf.unsqueeze(0),
        ).to(torch.int8)
        zeros = zeros + 1
        zeros = torch.bitwise_and(zeros, (2**bits) - 1)
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        #TODO: optimize: set zeros to int8
        zeros = zeros * scales
        zeros = zeros.reshape(-1, zeros.shape[2])
    else:
        zeros = None

    # weight rearrange
    if weight.shape[0] % rearrange_group != 0:
        padding = torch.zeros(
            [weight.shape[0] % rearrange_group, weight.shape[1]],
            dtype=weight.dtype,
            device=weight.device,
        )
        weight = torch.concat([weight, padding], dim=0)
    rweight_shape = (int(weight.shape[0] / 2), weight.shape[1])
    rweight = torch.zeros(rweight_shape, dtype=torch.uint8).to(qweight.device)
    half_group = int(rearrange_group / 2)
    try:
        shifts = torch.arange(0, weight.shape[0], device=qweight.device).reshape(
            int(weight.shape[0] / half_group), -1
        )
        rweight |= torch.bitwise_left_shift(weight[shifts[::2].reshape(-1)], 0)
        rweight |= torch.bitwise_left_shift(weight[shifts[1::2].reshape(-1)], 4)
    except Exception as e:
        raise RuntimeError(f"weight rearrange error: {e}")

    return rweight, zeros


def rearrange_uint4_int32_uint8_awq(
    method, qweight, qzeros, scales, rearrange_group=128, zeros_in_int8=False
):
    assert rearrange_group % 2 == 0, "rearrange group must be multiple of 2."
    qweight_shape = qweight.shape
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    shifts = torch.arange(0, 32, method.quant_config.weight_bits, device=qweight.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)
    assert (
        qweight_shape[1] * method.quant_config.pack_factor == iweights.shape[1]
    ), f"unpacked qweight shape error: {qweight_shape} and {iweights.shape}"
    # unpacking columnwise
    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    izeros = izeros.view(izeros.shape[0], -1)
    assert (
        qweight_shape[1] * method.quant_config.pack_factor == izeros.shape[1]
    ), f"unpacked qzeros shape error: {qweight_shape} and {izeros.shape}"
    reverse_order_tensor = torch.arange(
        iweights.shape[-1], dtype=torch.int32, device=qweight.device
    )
    reverse_order_tensor = reverse_order_tensor.view(
        -1, 32 // method.quant_config.weight_bits
    )
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)
    iweights = iweights[:, reverse_order_tensor]
    izeros = izeros[:, reverse_order_tensor]
    # overflow checks
    iweights = torch.bitwise_and(iweights, (2**method.quant_config.weight_bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**method.quant_config.weight_bits) - 1)

    # optimize: set zeros to int8
    if not zeros_in_int8:
        izeros = izeros * scales

    # weight rearrange
    if iweights.shape[0] % rearrange_group != 0:
        padding = torch.zeros(
            [iweights.shape[0] % rearrange_group, iweights.shape[1]],
            dtype=iweights.dtype,
            device=iweights.device,
        )
        iweights = torch.concat([iweights, padding], dim=0)
    rweight_shape = (int(iweights.shape[0] / 2), iweights.shape[1])
    rweight = torch.zeros(rweight_shape, dtype=torch.uint8).to(qweight.device)
    half_group = int(rearrange_group / 2)
    try:
        shifts = torch.arange(0, iweights.shape[0], device=qweight.device).reshape(
            int(iweights.shape[0] / half_group), -1
        )
        rweight |= torch.bitwise_left_shift(iweights[shifts[::2].reshape(-1)], 0)
        rweight |= torch.bitwise_left_shift(iweights[shifts[1::2].reshape(-1)], 4)
    except Exception as e:
        raise RuntimeError(f"weight rearrange error: {e}")
    if zeros_in_int8:
        # optimize: set zeros/weight to int8
        return rweight.to(torch.int8), izeros
    else:
        return rweight, izeros
