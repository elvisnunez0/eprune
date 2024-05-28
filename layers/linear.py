from argparse import ArgumentParser

from torch import nn

from layers import LAYER_REGISTRY


@LAYER_REGISTRY.register(name="linear", class_type="layer")
class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        return parser
