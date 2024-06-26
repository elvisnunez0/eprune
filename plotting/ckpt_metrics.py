import os
import re

import torch
from torch import Tensor


def get_ckpt_from_dir_name(folder_name: str) -> int:
    """
    Given the name of a checkpoint directory (e.g., 'ckpt40'),
    returns the checkpoint number (40 in this example).
    """
    match = re.search(r"ckpt(\d+)", folder_name)
    if match is not None:
        return int(match.group(1))
    else:
        return None


def get_best_top1(metrics: dict):
    top1_vals = [epoch_metrics["top1"] for epoch_metrics in metrics.values()]
    return max(top1_vals)


def get_accs_and_epochs(ckpts_dir: str, baseline_dir: str) -> None:
    """
    Given the directory @ckpts_dir containing the directories
        - ckpt20
        - ckpt40
        .
        .
        .
        - ckpt180

    gets the top accuracy from each folder and returns a two lists. One list
    contains the ckpt number and the other the corresponding best accuracy.

    Also returns the accuracy of the baseline model in the test_metrics.path file
    contrained in @baseline_dir.
    """

    # ckpt accs
    ckpt_nums = sorted(
        [get_ckpt_from_dir_name(dir_name) for dir_name in os.listdir(ckpts_dir)]
    )
    best_accs = []

    for ckpt_num in ckpt_nums:
        metrics_path = os.path.join(ckpts_dir, f"ckpt{ckpt_num}", "test_metrics.pth")
        metrics = torch.load(metrics_path, map_location="cpu")
        best_top1 = get_best_top1(metrics)
        best_accs.append(best_top1)

    # baseline acc
    baseline_metrics_path = os.path.join(baseline_dir, "test_metrics.pth")
    baseline_metrics = torch.load(baseline_metrics_path, map_location="cpu")
    best_baseline_acc = get_best_top1(baseline_metrics)

    return ckpt_nums, best_accs, best_baseline_acc


def get_accs_during_training(base_dir: str, metric: str = "top1") -> Tensor:
    metrics_path = os.path.join(base_dir, "test_metrics.pth")
    metrics = torch.load(metrics_path, map_location="cpu")

    # metrics is dict with keys 0, 1, ..., (epochs) and values
    # {"top1": <float>, "top5: <float>"}
    metrics = [metrics[epoch][metric] for epoch in range(len(metrics))]

    return metrics
