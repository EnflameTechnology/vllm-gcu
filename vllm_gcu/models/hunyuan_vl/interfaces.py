from typing import (ClassVar, Literal, Protocol, overload,
                    runtime_checkable)
from typing_extensions import TypeIs

import torch

@runtime_checkable
class SupportsXDRoPE(Protocol):
    """The interface required for all models that support XD-RoPE."""

    supports_xdrope: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports XD-RoPE.

    Note:
        There is no need to redefine this flag if this class is in the
        XDRope of your model class.
    """

    def get_xdrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list["MultiModalFeatureSpec"],
    ) -> torch.Tensor:
        """
        Get XD-RoPE input positions and delta value for this specific model.

        This method should be implemented by each model that supports XD-RoPE
        to provide model-specific logic for computing input positions.

        Args:
            input_tokens: List of input token IDs
            mm_features: Information about each multi-modal data item

        Returns:
            llm_positions: Tensor of shape `[xdrope_dim, num_tokens]` with
            4D(P/W/H/T) or 3D(W/H/T) positions.
        """
        ...


@overload
def supports_xdrope(model: type[object]) -> TypeIs[type[SupportsXDRoPE]]: ...


@overload
def supports_xdrope(model: object) -> TypeIs[SupportsXDRoPE]: ...


def supports_xdrope(
    model: type[object] | object,
) -> TypeIs[type[SupportsXDRoPE]] | TypeIs[SupportsXDRoPE]:
    return isinstance(model, SupportsXDRoPE)