from argparse import ArgumentParser

from torch import nn


class BaseLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        return parser
