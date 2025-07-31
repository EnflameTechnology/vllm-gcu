from unittest.mock import patch


def should_reinplace_scatter(node):
    return False


patch(
    "torch._inductor.fx_passes.reinplace.should_reinplace_scatter",
    should_reinplace_scatter,
).start()

from torch._inductor.fx_passes.reinplace import (  # noqa
    inplaceable_ops,
    _generalized_scatter,
    InplaceableOp,
    _inplace_generalized_scatter,
)

inplaceable_ops[_generalized_scatter] = InplaceableOp(
    _inplace_generalized_scatter,
    0,
    extra_check=should_reinplace_scatter,
)
