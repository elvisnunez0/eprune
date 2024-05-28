import os
from argparse import ArgumentParser, Namespace

import torch
from loguru._logger import Logger

from metrics.compose import MetricComposer
from models.base_model import BaseModel
from optim.optimizer.base_optimizer import BaseOptimizer
from saving.utils import create_dir, delete_file, isdir


class Saver(object):
    def __init__(
        self,
        cfg: Namespace,
        model: BaseModel,
        optimizer: BaseOptimizer,
        test_metrics: MetricComposer,
        logger: Logger,
    ) -> None:
        self.save_dir = getattr(cfg, "saver.dir")
        self.overwrite = getattr(cfg, "saver.overwrite")
        self.save_every_k_epochs = getattr(cfg, "saver.save_every_k_epochs")
        self.model_dict_only = getattr(cfg, "saver.model_dict_only")
        self.best_metric = getattr(cfg, "saver.best_metric")
        self.max_epochs = getattr(cfg, "optim.lr_scheduler.max_epochs")
        self.best_metric_max = getattr(cfg, "saver.best_metric_max")

        self._create_dirs()

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.test_metrics = test_metrics
        self.logger = logger
        self.best_metric_value = None

        self.running_test_metrics_dict = {}

    def _create_dirs(self):
        # Check if overwriting is allowed
        if isdir(self.save_dir) and not self.overwrite:
            msg = (
                f"The save directory {self.save_dir} exists. "
                "Set --saver.overwrite to True to overwrite."
            )
            raise IsADirectoryError(msg)

        # Directory where model and optimizer will be saved
        self.model_dir = os.path.join(self.save_dir, "model")
        create_dir(self.save_dir)
        create_dir(self.model_dir)

    def improved_metric(self, current_metric_value: float) -> bool:
        if self.best_metric_max:
            return current_metric_value > self.best_metric_value
        else:
            return current_metric_value < self.best_metric_value

    def save(self, epoch: int):
        # Get metrics
        test_metrics_dict = self.test_metrics.get_metric_dict()

        # Save model and optimizer
        model_artifact = self.model.state_dict() if self.model_dict_only else self.model
        model_save_dict = {
            "model": model_artifact,
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        current_metric_value = test_metrics_dict[self.best_metric]
        model_best_path = os.path.join(
            self.model_dir, f"best_ckpt__{current_metric_value:0.2f}.pth"
        )
        # We always save the best metric
        if self.best_metric_value is None or self.improved_metric(current_metric_value):
            torch.save(model_save_dict, model_best_path)
            self.logger.info(f"Saved best model to {model_best_path}.")
            if self.best_metric_value is not None:
                prev_best_path = os.path.join(
                    self.model_dir, f"best_ckpt__{self.best_metric_value:0.2f}.pth"
                )
                if delete_file(prev_best_path):
                    self.logger.info(
                        f"Deleted previous best model at {prev_best_path}."
                    )
                else:
                    self.logger.info(f"Failed to delete previous best model.")
            self.best_metric_value = current_metric_value

        # Optionally, save the checkpoint evey multiple of k epochs
        model_path = os.path.join(self.model_dir, f"ckpt{epoch}.pth")
        if self.save_every_k_epochs > 0 and epoch % self.save_every_k_epochs == 0:
            torch.save(model_save_dict, model_path)
            self.logger.info(f"Saved model checkpoint to {model_path}.")
        elif epoch == self.max_epochs:
            # Save final epoch
            torch.save(model_save_dict, model_path)
            self.logger.info(f"Saved final model checkpoint to {model_path}.")

        # Save test metrics
        metrics_path = os.path.join(self.save_dir, f"test_metrics.pth")
        self.running_test_metrics_dict[epoch] = test_metrics_dict
        torch.save(self.running_test_metrics_dict, metrics_path)
        self.logger.info(f"Saved test metrics to {metrics_path}.")

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--saver.dir",
            type=str,
            default="reuslts",
            help="The directory in which to save training artifacts. Defaults to results.",
        )

        group.add_argument(
            "--saver.overwrite",
            action="store_true",
            default=False,
            help="If the save directory exists, will overwrite existing files. Defaults to False.",
        )

        group.add_argument(
            "--saver.save-every-k-epochs",
            type=int,
            default=-1,
            help="Whether to save the model every multiple of k epochs. Defaults to -1 (don't save every epoch).",
        )

        group.add_argument(
            "--saver.model-dict-only",
            action="store_true",
            default=True,
            help="Whether to only save the model's state dict. Defaults to True.",
        )

        group.add_argument(
            "--saver.best-metric",
            type=str,
            default="top1",
            help="Specifies which metric to use to determine the best model to save. Defaults to top1.",
        )

        group.add_argument(
            "--saver.best-metric-max",
            action="store_true",
            default=True,
            help="Specifies whether the best metric should be maximized. Defaults to True.",
        )

        return parser
