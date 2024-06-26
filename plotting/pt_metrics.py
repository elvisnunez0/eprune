import os
from typing import List

import torch


def get_pt_metrics(pt_metrics_dir: str) -> List:
    metrics_path = os.path.join(pt_metrics_dir, "metrics.pth")
    metrics = torch.load(metrics_path)
    metrics = [metrics[epoch] for epoch in range(len(metrics))]

    return metrics
