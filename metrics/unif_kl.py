from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru._logger import Logger
from tqdm import tqdm

from metrics import METRIC_REGISTRY
from metrics.post_train_metric import PostTrainMetric


@METRIC_REGISTRY.register("uniform_kl", class_type="metric")
class UniformKL(PostTrainMetric):
    def __init__(self, cfg: Namespace, device: torch.device, logger: Logger) -> None:
        super().__init__(cfg=cfg, device=device, logger=logger)

        self.num_classes = getattr(cfg, "model.classification.num_classes")
        self.mc_epochs = getattr(
            cfg,
            "metrics.post_train.entropy.mc_epochs",
        )

    def compute_uniform_kl(self, model: nn.Module):
        """ """

        # Prepare model
        if self.recalibrate_bn:
            self.recalibrate_model(model, self.data_loader)

        # Put model in eval
        model.eval()

        avg_neg_log_sum = 0.0
        num_batches = 0

        with torch.no_grad():
            for k in range(self.mc_epochs):
                for i, batch in enumerate(
                    tqdm(self.data_loader, leave=False, desc="Computing entropy")
                ):
                    data = batch["sample"].to(self.device)
                    output = model(data)
                    pred_prob = F.softmax(output, dim=-1)  # [B, num classes]
                    neg_log_sum = -(torch.log2(pred_prob)).sum(dim=-1)
                    avg_batch_neg_log_sum = neg_log_sum.mean(dim=0)

                    avg_neg_log_sum += avg_batch_neg_log_sum.item()
                    num_batches += 1

        unif_kl = -torch.log2(torch.tensor(self.num_classes)) + avg_neg_log_sum / (
            num_batches * self.mc_epochs * self.num_classes
        )

        return unif_kl

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--metrics.post-train.uniform-kl.mc-epochs",
            type=int,
            default=1,
            help="The number of epochs to use in the MC sampling. Defaults to 1.",
        )

        return parser
