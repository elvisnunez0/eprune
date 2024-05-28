import copy
import os
import re
from argparse import Namespace
from typing import Optional

import torch.nn as nn
from torch.nn import DataParallel

from args.config import get_arguments
from logger import configure_logger, logger
from metrics.entropy import ConditionalEntropy
from models import get_model_from_registry
from models.pruners.ustruct_pruner import UStructPruner
from saving.entropy_saver import EntropySaver
from utils.device import get_device


def num_ckpts_in_dir(dir: str) -> int:
    """
    Returns the number of files of the form ckpt<epoch>.pth are
    in the provided directory.
    """
    ckpt_count = 0
    pattern = re.compile(r"ckpt\d+\.pth")

    for f in os.listdir(dir):
        if pattern.match(f):
            ckpt_count += 1

    return ckpt_count


def get_model(
    cfg: Namespace, device: str, ckpt_num: int, model_ckpts_dir: str
) -> nn.Module:
    cfg_cp = copy.deepcopy(cfg)
    ckpt_path = os.path.join(model_ckpts_dir, f"ckpt{ckpt_num}.pth")
    setattr(cfg_cp, "model.load_from_state_dict.path", ckpt_path)

    model = get_model_from_registry(cfg=cfg_cp)

    # Set distributed training
    data_parallel = getattr(cfg, "common.distributed_training.data_parallel")
    model = model.to(device=device)

    if device == "cuda" and data_parallel:
        model = DataParallel(model)

    # Set pruning module
    uprune_enable = getattr(cfg, "model.sparsity.unstructured.enable")
    if uprune_enable:
        pruner = UStructPruner(cfg=cfg, model=model, logger=logger, device=device)

        sparsity = pruner.compute_sparsity_from_mask()
        logger.info(f"Set model (unstructured) sparsity to {sparsity:0.4f}.")

    return model


def main(cfg: Optional[Namespace] = None):
    if cfg is None:
        cfg = get_arguments(update_from_config=True)

    # Set logger
    configure_logger(cfg)

    device = get_device()
    entropy = ConditionalEntropy(cfg=cfg, device=device, logger=logger)

    model_ckpts_dir = getattr(cfg, "metrics.post_train.model_ckpts_dir")
    num_ckpts = num_ckpts_in_dir(model_ckpts_dir)

    entropies = {}

    entropy_saver = EntropySaver(cfg, logger)

    for ckpt_num in range(num_ckpts):
        model = get_model(
            cfg=cfg, device=device, ckpt_num=ckpt_num, model_ckpts_dir=model_ckpts_dir
        )
        model_entropy = entropy.compute_entropy(model=model)
        entropies[ckpt_num] = model_entropy

        entropy_saver.save(metrics=entropies)


if __name__ == "__main__":
    cfg = get_arguments()
    main(cfg=cfg)
