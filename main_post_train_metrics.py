import copy
import os
import re
from argparse import Namespace
from typing import Dict, Optional

import torch
import torch.nn as nn
from loguru._logger import Logger
from torch.nn import DataParallel

from args.config import get_arguments
from logger import configure_logger, logger
from metrics.entropy import ConditionalEntropy
from metrics.fisher import Fisher
from metrics.unif_kl import UniformKL
from models import get_model_from_registry
from models.pruners.ustruct_pruner import UStructPruner
from saving.post_train_metrics_saver import PostTrainMetricsSaver
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


def compute_fisher_metrics(
    cfg: Namespace, device: torch.device, logger: Logger
) -> Dict:
    fisher = Fisher(cfg=cfg, device=device, logger=logger)
    fisher_saver = PostTrainMetricsSaver(cfg, logger)
    model_ckpts_dir = getattr(cfg, "metrics.post_train.model_ckpts_dir")
    num_ckpts = num_ckpts_in_dir(model_ckpts_dir)

    per_layer_fisher_traces = {}

    for ckpt_num in range(num_ckpts):
        model = get_model(
            cfg=cfg, device=device, ckpt_num=ckpt_num, model_ckpts_dir=model_ckpts_dir
        )
        per_layer_fisher = fisher.compute_per_layer_fisher(model=model)
        per_layer_fisher_traces[ckpt_num] = per_layer_fisher

        fisher_saver.save(metrics=per_layer_fisher_traces)


def compute_entropy_metrics(
    cfg: Namespace, device: torch.device, logger: Logger
) -> Dict:
    entropy = ConditionalEntropy(cfg=cfg, device=device, logger=logger)

    model_ckpts_dir = getattr(cfg, "metrics.post_train.model_ckpts_dir")
    num_ckpts = num_ckpts_in_dir(model_ckpts_dir)

    entropies = {}

    entropy_saver = PostTrainMetricsSaver(cfg, logger)

    for ckpt_num in range(num_ckpts):
        model = get_model(
            cfg=cfg, device=device, ckpt_num=ckpt_num, model_ckpts_dir=model_ckpts_dir
        )
        model_entropy = entropy.compute_entropy(model=model)
        entropies[ckpt_num] = model_entropy

        entropy_saver.save(metrics=entropies)


def compute_uniform_kl_metrics(
    cfg: Namespace, device: torch.device, logger: Logger
) -> Dict:
    unif_kl = UniformKL(cfg=cfg, device=device, logger=logger)

    model_ckpts_dir = getattr(cfg, "metrics.post_train.model_ckpts_dir")
    num_ckpts = num_ckpts_in_dir(model_ckpts_dir)

    uniform_kl_vals = {}

    kl_saver = PostTrainMetricsSaver(cfg, logger)

    for ckpt_num in range(num_ckpts):
        model = get_model(
            cfg=cfg, device=device, ckpt_num=ckpt_num, model_ckpts_dir=model_ckpts_dir
        )
        model_unif_kl = unif_kl.compute_uniform_kl(model=model)
        uniform_kl_vals[ckpt_num] = model_unif_kl

        kl_saver.save(metrics=uniform_kl_vals)


def main(cfg: Optional[Namespace] = None):
    if cfg is None:
        cfg = get_arguments(update_from_config=True)
    configure_logger(cfg)
    device = get_device()

    metrics_name = getattr(cfg, "metrics.post_train.metric_name")
    if metrics_name is None:
        raise ValueError(
            "metrics.post_train.metric_name must be specified. One of 'fisher' or 'entropy'."
        )

    if metrics_name == "fisher":
        compute_fisher_metrics(cfg=cfg, device=device, logger=logger)
    elif metrics_name == "entropy":
        compute_entropy_metrics(cfg=cfg, device=device, logger=logger)
    elif metrics_name == "uniform_kl":
        compute_uniform_kl_metrics(cfg=cfg, device=device, logger=logger)


if __name__ == "__main__":
    cfg = get_arguments()
    main(cfg=cfg)
