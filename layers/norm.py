from torch import nn

from layers import LAYER_REGISTRY
from layers.base_layer import BaseLayer


@LAYER_REGISTRY.register(name="batchnorm1d", class_type="layer")
class BatchNorm1d(BaseLayer, nn.BatchNorm1d):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )


@LAYER_REGISTRY.register(name="batchnorm2d", class_type="layer")
class BatchNorm2d(BaseLayer, nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
