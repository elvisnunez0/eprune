import os
from argparse import Namespace

import yaml


def isdir(dir: str) -> None:
    """
    Checks if the specified directory exists.

    Args:
        dir: The directory to check.
    """
    return os.path.isdir(dir)


def create_dir(dir: str) -> None:
    """
    Creates the specified folder if it does not already exist.

    Args:
        dir: The directory to create.
    """
    if not isdir(dir):
        os.makedirs(dir)


def delete_file(path: str) -> None:
    try:
        os.remove(path)
        return 1
    except Exception as e:
        return -1


def save_cfg_to_yaml(cfg: Namespace, savedir: str):
    # TODO: Will this save python types to the config file?
    save_path = os.path.join(savedir, "config.yaml")
    with open(save_path, "w") as f:
        yaml.dump(cfg, f)
