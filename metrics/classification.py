import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Union

import torch
import torch.nn.functional as F
from loguru._logger import Logger
from torch import Tensor

from metrics import METRIC_REGISTRY
from metrics.base_metric import BaseMetric
from saving.utils import create_dir


@METRIC_REGISTRY.register("accuracy", class_type="metric")
class Accuracy(BaseMetric):
    def __init__(self, cfg: Namespace, mode: str) -> None:
        super().__init__(cfg)

        metric_names = sorted(getattr(cfg, f"metrics.{mode}"))

        self.running_sum_top1 = 0
        self.running_sum_top5 = 0
        self.seen_samples = 0

        self.top1 = "top1" in metric_names
        self.top5 = "top5" in metric_names

    def reset(self) -> None:
        self.running_sum_top1 = 0.0
        self.running_sum_top5 = 0.0
        self.seen_samples = 0

    def save(self, logger: Logger) -> None:
        return

    @torch.no_grad()
    def update(self, prediction: Union[Dict, Tensor], target: Tensor) -> None:
        if isinstance(prediction, Dict):
            logits = prediction["logits"]
        else:
            logits = prediction

        assert logits.dim() == 2, "Logits shape must be [B, classes]."

        B = logits.shape[0]
        if self.top1:
            top1_correct = torch.argmax(logits, dim=1).eq(target).sum().item()
            self.running_sum_top1 += top1_correct
        if self.top5:
            target = target.view(-1, 1)
            _, top5_indices = torch.topk(logits, k=5, dim=1)
            top5_correct = top5_indices.eq(target).any(dim=1).sum().item()
            self.running_sum_top5 += top5_correct
        self.seen_samples += B

    def get_metrics(self):
        if self.seen_samples == 0:
            # This is typically the case at initialization. Since we are not
            # really interested in performance at this checkpoint, just return 0.
            top1 = 0.0
            top5 = 0.0
        else:
            top1 = 100.0 * self.running_sum_top1 / self.seen_samples
            top5 = 100.0 * self.running_sum_top5 / self.seen_samples

        return {"top1": top1, "top5": top5}


@METRIC_REGISTRY.register("cross_entropy", class_type="metric")
class CrossEntropy(BaseMetric):
    def __init__(self, cfg: Namespace, mode: str) -> None:
        super().__init__(cfg)

        self.running_sum = 0
        self.seen_samples = 0

    def reset(self) -> None:
        self.running_sum = 0.0
        self.seen_samples = 0

    def save(self, logger: Logger) -> None:
        return

    @torch.no_grad()
    def update(self, prediction: Union[Dict, Tensor], target: Tensor) -> None:
        if isinstance(prediction, Dict):
            logits = prediction["logits"]
        else:
            logits = prediction

        assert logits.dim() == 2, "Logits shape must be [B, classes]."

        B = logits.shape[0]
        cross_entropy = F.cross_entropy(logits, target, reduction="sum").item()
        self.running_sum += cross_entropy
        self.seen_samples += B

    def get_metrics(self):
        return {
            "cross_entropy": self.running_sum / self.seen_samples,
        }


@METRIC_REGISTRY.register("sphere_projection", class_type="metric")
class SphereProjection(BaseMetric):
    def __init__(self, cfg: Namespace, mode: str) -> None:
        super().__init__(cfg)

        self.savedir = getattr(cfg, "metrics.sphere_projection.savedir")

        if self.savedir is None:
            saver_savedir = getattr(cfg, "saver.dir")
            self.savedir = os.path.join(saver_savedir, "eval")
        create_dir(self.savedir)

        self.projections = None
        self.targets = None

    def reset(self) -> None:
        self.projections = None
        self.targets = None

    @torch.no_grad()
    def update(self, prediction: Union[Dict, Tensor], target: Tensor) -> None:
        sphere_projections = prediction["sphere_projection"]
        if self.projections is None:
            self.projections = sphere_projections
            self.targets = target
        else:
            self.projections = torch.cat([self.projections, sphere_projections])
            self.targets = torch.cat([self.targets, target])

    def save(self, logger: Logger) -> None:
        savedict = {"projections": self.projections, "targets": self.targets}
        savepath = os.path.join(self.savedir, "sphere_projections.pth")
        torch.save(savedict, savepath)

        logger.info(f"Saved spherical projections to {savepath}.")

    def get_metrics(self):
        return {}

    @classmethod
    def add_arguments(cls, parser: ArgumentParser):
        if cls != SphereProjection:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--metrics.sphere-projection.savedir",
            type=str,
            default=None,
            help="The directory in which to save the spherical projections. Defaults to None.",
        )

        return parser
