import os
from argparse import Namespace
from typing import Dict

import torch
from loguru._logger import Logger

from saving.utils import create_dir, save_cfg_to_yaml


class EntropySaver(object):
    def __init__(
        self,
        cfg: Namespace,
        logger: Logger,
    ) -> None:
        self.save_dir = getattr(cfg, "saver.dir")
        self.metrics_path = os.path.join(self.save_dir, "metrics.pth")

        create_dir(self.save_dir)

        self.cfg = cfg
        self.logger = logger

        save_cfg_to_yaml(self.cfg, savedir=self.save_dir)

    def save(self, metrics: Dict):
        torch.save(metrics, self.metrics_path)
        self.logger.info(f"Saved entropy metrics to {self.metrics_path}.")
