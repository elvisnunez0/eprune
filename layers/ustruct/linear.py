import torch.nn.functional as F
from torch import Tensor, nn

from layers import LAYER_REGISTRY
from layers.ustruct.base_ustruct_layer import BasePruneModule


@LAYER_REGISTRY.register(name="ustruct_linear", class_type="layer")
class UStructLinear(nn.Linear, BasePruneModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        prune_bias: bool = False,
        sparsity_scale: float = 1.0,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        BasePruneModule.__init__(
            self, prune_bias=prune_bias, sparsity_scale=sparsity_scale
        )

        self.register_mask_buffers()

    def forward(self, input: Tensor) -> Tensor:
        w_ = self.pruned_weight()
        b_ = self.pruned_bias()

        return F.linear(input, weight=w_, bias=b_)

    def __repr__(self):
        sparsity_rate = self.get_sparsity_rate()
        bias = True if self.bias is not None else False
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={bias}, "
            f"sparsity_rate={sparsity_rate:0.2f}, "
            f"prune_bias={self.prune_bias})"
        )
