from typing import Union

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from layers import LAYER_REGISTRY
from layers.ustruct.base_ustruct_layer import BasePruneModule


@LAYER_REGISTRY.register(name="ustruct_conv2d", class_type="layer")
class UStructConv2d(nn.Conv2d, BasePruneModule):
    """
    Conv2d layer pruned via unstructured pruning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[_size_2_t, str] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        prune_bias: bool = False,
        sparsity_scale: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        BasePruneModule.__init__(
            self, prune_bias=prune_bias, sparsity_scale=sparsity_scale
        )
        self.register_mask_buffers()

    def forward(self, input: Tensor) -> Tensor:
        w_ = self.pruned_weight()
        b_ = self.pruned_bias()

        return F.conv2d(
            input,
            weight=w_,
            bias=b_,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def __repr__(self):
        sparsity_rate = self.get_sparsity_rate()
        bias = True if self.bias is not None else False
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"bias={bias}, "
            f"sparsity_rate={sparsity_rate:0.2f}, "
            f"prune_bias={self.prune_bias})"
        )
