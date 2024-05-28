from argparse import ArgumentParser

from saving.saver import Saver


def get_saver_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = Saver.add_arguments(parser)
    return parser
