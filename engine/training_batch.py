import time
from argparse import Namespace

import torch
from loguru._logger import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss.base_loss import BaseLoss
from metrics.compose import MetricComposer
from metrics.fisher import Fisher
from models.base_model import BaseModel
from optim.lr_scheduler.base_lr_scheduler import BaseLRScheduler
from optim.optimizer.base_optimizer import BaseOptimizer
from saving import Saver


class Trainer(object):
    def __init__(
        self,
        cfg: Namespace,
        model: BaseModel,
        optimizer: BaseOptimizer,
        lr_scheduler: BaseLRScheduler,
        criteria: BaseLoss,
        start_epoch: int,
        logger: Logger,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        validation_loader: DataLoader = None,
        train_loader_no_aug: DataLoader = None,
        callbacks=None,
    ) -> None:
        self.log_freq = getattr(cfg, "common.logger.frequency")
        self.max_epochs = getattr(cfg, "optim.lr_scheduler.max_epochs")

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criteria = criteria
        self.start_epoch = start_epoch
        self.logger = logger
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.train_loader_no_aug = train_loader_no_aug
        self.callbacks = callbacks

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        batch_indices = []

        self.logger.info(f"Starting epoch {epoch}")
        for batch_id, batch in enumerate(self.train_loader):
            indices = list(batch["sample_id"].detach().cpu().numpy())
            batch_indices = batch_indices + indices

        return batch_indices

    def run(self) -> None:
        batch_indices = []
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Set epoch in sampler
            # self.train_loader.batch_sampler.set_epoch(epoch)

            # Train
            epoch_indices = self.train_epoch(epoch)

            batch_indices.append(epoch_indices)

        return batch_indices
