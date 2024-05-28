from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from loguru._logger import Logger

from data.loaders import create_loader


class PostTrainMetric(object):

    def __init__(self, cfg: Namespace, device: torch.device, logger: Logger) -> None:
        self.cfg = cfg
        self.device = device
        self.logger = logger

        self.use_training_augs = getattr(cfg, "metrics.post_train.use_training_augs")
        self.eval_mode = getattr(cfg, "metrics.post_train.eval_mode")
        self.recalibrate_bn = getattr(cfg, "metrics.post_train.recalibrate_bn")

        # Set data loader
        data_loader_mode = "train" if self.use_training_augs else "train_no_aug"
        self.data_loader = create_loader(cfg, mode=data_loader_mode, shuffle=True)

        # Set loss function
        self._set_loss_fn()

    def _set_loss_fn(self) -> None:
        loss_fn_category = getattr(self.cfg, "loss.category")
        loss_fn_name = getattr(self.cfg, f"loss.{loss_fn_category}.name")

        if loss_fn_name == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError()

    def recalibrate_model(self, model: nn.Module, data_loader) -> nn.Module:
        self.logger.info(f"Recalibrating model...")
        # Set to train mode to allow BN running stats to update
        model.train()

        for data, _ in data_loader:
            model(data)
        self.logger.info(f"Done calibrating...\n")

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        if cls != PostTrainMetric:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--metrics.post-train.metric-name",
            type=str,
            help="The name of the post-train metric to compute. Defaults to None.",
        )

        group.add_argument(
            "--metrics.post-train.use-training-augs",
            action="store_true",
            help="Whether the data loader used to compute the post-train metrics should use "
            "training augmentations. Defaults to False.",
        )

        group.add_argument(
            "--metrics.post-train.eval-mode",
            action="store_true",
            default=True,
            help="Whether the model should put in eval mode prior to computing post-train metrics. "
            "Defaults to True.",
        )

        group.add_argument(
            "--metrics.post-train.recalibrate-bn",
            action="store_true",
            help="Whether to recalibrate the BN stats prior to computing the post-train metrics. "
            "Defaults to False.",
        )

        group.add_argument(
            "--metrics.post-train.model-ckpts-dir",
            type=str,
            help="The directory containing model checkpoints for which the post-train metric "
            "will be computed. Defaults to None.",
        )

        return parser
