import os

from saving.utils import create_dir


def create_save_dir_from_path(path: str):
    """
    Given the path to a file (typically a save destination),
    creates the directory in case it does not exist.
    """
    directory, filename = os.path.split(path)
    create_dir(directory)
