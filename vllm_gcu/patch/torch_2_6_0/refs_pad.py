import torch
from torch._refs import register_decomposition, aten, out_wrapper
import builtins
import torch._prims as prims
import torch._prims_common as utils
from torch._prims_common import NumberType, TensorLikeType
from typing import List
from torch._decomp import decomposition_table, remove_decompositions

remove_decompositions(decomposition_table, [aten.constant_pad_nd])


@register_decomposition(aten.constant_pad_nd)
@out_wrapper()
def constant_pad_nd(
    input: TensorLikeType, pad: List[int], value: NumberType = 0
) -> TensorLikeType:
    torch._check(
        len(pad) % 2 == 0,
        lambda: f"Length of pad must be even but instead it equals {len(pad)}",
    )

    input_sizes = input.shape
    l_inp = len(input_sizes)

    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad

    torch._check(
        l_inp >= l_pad,
        lambda: "Length of pad should be no more than twice the number of "
        f"dimensions of the input. Pad length is {len(pad)} while the input has "
        f"{l_inp} dimensions.",
    )

    c_input = input
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        if pad[pad_idx] < 0:
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.shape[i] + pad[pad_idx])

        if pad[pad_idx + 1] < 0:
            c_input = c_input.narrow(i, 0, c_input.shape[i] + pad[pad_idx + 1])

    # If all the pads are negative we can return the result.
    # Avoid early exiting if all pads = 0 to prevent specialization on export.
    # During export, raw if statements are specialized on the input, meaning
    # that we lose a branch depending on the example input used to export.
    # Here, this is either the case where all pads = 0, or the case where at
    # least one pad > 0 and the rest are >= 0.
    # Avoiding the early exit when all pads = 0 ensures we can export
    # constant_pad_nd for cases when all pads >= 0.
    # Note: if any pads are negative, this code specializes due to the if statements above.
    if builtins.all(p < 0 for p in pad):
        return c_input.clone()

    new_shape = list(input_sizes[:l_diff])

    for i in range(l_pad):
        pad_idx = len(pad) - ((i + 1) * 2)
        new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1]
        torch._check(
            new_dim >= 0,
            lambda: f"The input size {input_sizes[l_diff + i]}, plus negative padding "
            f"{pad[pad_idx]} and {pad[pad_idx + 1]} resulted in a negative output size, "
            f"which is invalid. Check dimension {l_diff + i} of your input.",
        )
        new_shape.append(new_dim)

    memory_format = utils.suggest_memory_format(input)
    output = torch.empty(
        new_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
        memory_format=memory_format,
    )

    if value == 0 and input.dtype == torch.bool:
        value = False
    # torch.fill isn't typed to allow complex values
    output = torch.fill(output, value)  # type: ignore[arg-type]

    c_output = output
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        if pad[pad_idx] >= 0:
            c_output = c_output.narrow(
                i, pad[pad_idx], c_output.shape[i] - pad[pad_idx]
            )
        if pad[pad_idx + 1] >= 0:
            c_output = c_output.narrow(i, 0, c_output.shape[i] - pad[pad_idx + 1])

    prims.copy_to(c_output, c_input)
    return output


# NOTE: this is not how original decomposition get dispatched,
# see FakeTensorMode._dispatch_impl for details.
constant_pad_nd.__module__ = 'torch._decomp'
setattr(torch._decomp.decompositions, 'constant_pad_nd', constant_pad_nd)
