from torch import nn

from layers import LAYER_REGISTRY
from layers.base_layer import BaseLayer


@LAYER_REGISTRY.register(name="relu", class_type="layer")
class ReLU(BaseLayer, nn.ReLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)
