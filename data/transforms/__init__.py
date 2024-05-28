from argparse import ArgumentParser

from utils.registry import Registry

TRANSFORMS_REGISTRY = Registry(
    name="transforms",
    subdirs=["data/transforms"],
)


def get_transform_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = TRANSFORMS_REGISTRY.get_all_arguments(parser)
    return parser
