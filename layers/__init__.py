from argparse import ArgumentParser

from utils.registry import Registry

LAYER_REGISTRY = Registry(
    name="layers",
    subdirs=["layers"],
)


def get_layer_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = LAYER_REGISTRY.get_all_arguments(parser)
    return parser
