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

        self.test_metrics = MetricComposer(cfg, mode="test")
        self.saver = Saver(
            cfg=self.cfg,
            model=self.model,
            optimizer=self.optimizer,
            test_metrics=self.test_metrics,
            logger=self.logger,
        )
        # self.fisher = Fisher(
        #     cfg=self.cfg,
        #     model=self.model,
        #     # loader=self.train_loader_no_aug,
        #     loader=self.train_loader,
        #     device=self.device,
        #     logger=self.logger,
        # )

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        total_loss = 0
        total_samples = 0
        n = len(self.train_loader)

        epoch_start_time = time.time()
        for batch_id, batch in enumerate(self.train_loader):
            sample = batch["sample"].to(self.device)
            target = batch["target"].to(self.device)
            B = sample.shape[0]

            self.optimizer.zero_grad()

            self.optimizer = self.lr_scheduler.update_lr(
                optimizer=self.optimizer, epoch=epoch
            )

            prediction = self.model(sample)
            loss = self.criteria(prediction, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_samples += B
            if batch_id % self.log_freq == 0:
                lr = self.lr_scheduler.get_lr(epoch)
                avg_loss = total_loss / (batch_id + 1)
                self.logger.info(
                    f"Epoch {epoch} ({total_samples}/{n}): Avg loss={avg_loss:0.4f} | LR={lr:0.4f}"
                )

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback()

        epoch_time = time.time() - epoch_start_time
        self.logger.info(f"Epoch {epoch} time: {epoch_time:0.2f}s")

    def eval_epoch(self) -> None:
        self.model.eval()
        self.logger.info("Testing...")
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), colour="cyan", unit="task") as bar:
                for batch_id, batch in enumerate(self.test_loader):
                    sample = batch["sample"].to(self.device)
                    target = batch["target"].to(self.device)
                    B = sample.shape[0]

                    prediction = self.model(sample)
                    self.test_metrics.update(prediction=prediction, target=target)

                    bar.update(B)
            self.test_metrics.log_metrics(logger=self.logger, prefix="Test")

    def clear_metrics_cache(self):
        """
        This function will clear the train/test caches. Typically called at the
        end of each epoch after metrics have been saved/logged.
        """
        self.test_metrics.reset()

    def run(self) -> None:
        # Save model checkpoint at initialization
        self.saver.save(0)

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Set epoch in sampler
            # self.train_loader.batch_sampler.set_epoch(epoch)

            # Train
            self.train_epoch(epoch)

            # Metrics such as Fisher
            # self.fisher.compute_metrics(save=True)

            # Eval
            self.eval_epoch()

            # Save model and metrics
            self.saver.save(epoch)
            self.clear_metrics_cache()
