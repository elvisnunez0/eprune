from argparse import Namespace
from typing import Any

import torch
from loguru._logger import Logger

from utils.device import get_device


def load_from_torch(pth: str) -> Any:
    device = get_device()
    if device == "cpu":
        return torch.load(pth, map_location="cpu")
    else:
        return torch.load(pth)


def get_start_epoch(cfg: Namespace, logger: Logger) -> int:
    # Check if we are loading the model from a checkpoint
    ckpt_path = getattr(cfg, "model.load_from_state_dict.path")
    resume_epoch = getattr(cfg, "model.load_from_state_dict.resume_training")
    if ckpt_path is not None and resume_epoch:
        ckpt = load_from_torch(ckpt_path)
        # +1 because the checkpoint is when the ckpt was
        # saved, so we continue at the next epoch.
        epoch = ckpt["epoch"] + 1
    else:
        epoch = 1

    logger.info(f"Setting start epoch to {epoch}.")
    return epoch
