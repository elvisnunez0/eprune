from argparse import Namespace

import torch
from loguru._logger import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss.base_loss import BaseLoss
from metrics.compose import MetricComposer
from metrics.fisher import Fisher
from models.base_model import BaseModel


class Evaluator(object):
    def __init__(
        self,
        cfg: Namespace,
        model: BaseModel,
        criteria: BaseLoss,
        logger: Logger,
        device: torch.device,
        loader: DataLoader,
    ) -> None:
        self.log_freq = getattr(cfg, "common.logger.frequency")

        self.cfg = cfg
        self.model = model
        self.criteria = criteria
        self.logger = logger
        self.device = device
        self.loader = loader

        self.test_metrics = MetricComposer(cfg, mode="test")
        self.fisher = Fisher(
            cfg=self.cfg,
            model=self.model,
            loader=self.loader,
            device=self.device,
            logger=self.logger,
        )

    def eval_epoch(self) -> None:
        self.model.eval()
        self.logger.info("Testing...")
        with torch.no_grad():
            with tqdm(total=len(self.loader), colour="cyan", unit="task") as bar:
                for batch_id, batch in enumerate(self.loader):
                    sample = batch["sample"].to(self.device)
                    target = batch["target"].to(self.device)
                    B = sample.shape[0]

                    prediction = self.model(sample)
                    self.test_metrics.update(prediction=prediction, target=target)

                    bar.update(B)
            self.test_metrics.log_metrics(logger=self.logger, prefix="Test")
            self.test_metrics.save(logger=self.logger)

    def clear_metrics_cache(self):
        """
        This function will clear the train/test caches. Typically called at the
        end of each epoch after metrics have been saved/logged.
        """
        self.test_metrics.reset()

    def model_sparsity(self):
        num_zeros = 0
        num_elem = 0
        for param in self.model.parameters():
            num_elem += param.numel()
            num_zeros += torch.eq(param, 0).sum()
        self.logger.info(f"model sparsity @eval is {num_zeros / num_elem}")

    def run(self) -> None:
        # fim_trace = self.fisher.compute_trace()
        # self.logger.info(f"tr(F) = {fim_trace:0.4f}")

        self.eval_epoch()
        self.clear_metrics_cache()
