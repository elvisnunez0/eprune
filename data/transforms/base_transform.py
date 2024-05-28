from argparse import ArgumentParser, Namespace


class BaseTransform(object):
    """
    Base class for data augmentations.
    """

    def __init__(self, cl_args: Namespace, *args, **kwargs) -> None:
        self.cl_args = cl_args

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        return parser
