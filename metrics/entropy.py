from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru._logger import Logger
from tqdm import tqdm

from metrics import METRIC_REGISTRY
from metrics.post_train_metric import PostTrainMetric


@METRIC_REGISTRY.register("entropy", class_type="metric")
class ConditionalEntropy(PostTrainMetric):
    def __init__(self, cfg: Namespace, device: torch.device, logger: Logger) -> None:
        super().__init__(cfg=cfg, device=device, logger=logger)

        self.mc_epochs = getattr(
            cfg,
            "metrics.post_train.entropy.mc_epochs",
        )

    def compute_entropy(self, model: nn.Module):
        """
        Computes H(Y|X) = E_{x ~ data distribution}[E_{y ~ model(x)}[H(Y|X=x)]].
        """

        # Prepare model
        if self.recalibrate_bn:
            self.recalibrate_model(model, self.data_loader)

        # Put model in eval
        model.eval()

        avg_entropy_sum = 0.0
        num_batches = 0

        with torch.no_grad():
            for k in range(self.mc_epochs):
                for i, batch in enumerate(
                    tqdm(self.data_loader, leave=False, desc="Computing entropy")
                ):
                    data = batch["sample"].to(self.device)
                    output = model(data)
                    pred_prob = F.softmax(output, dim=-1)  # [B, num classes]
                    # H(X) = -sum_x p(x)*log2(p(x))
                    entropy = -(pred_prob * torch.log2(pred_prob)).sum(dim=-1)
                    avg_batch_entropy = entropy.mean(dim=0)

                    avg_entropy_sum += avg_batch_entropy.item()
                    num_batches += 1

        avg_entropy = avg_entropy_sum / (num_batches * self.mc_epochs)

        return avg_entropy

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--metrics.post-train.entropy.mc-epochs",
            type=int,
            default=1,
            help="The number of epochs to use in the MC sampling. Defaults to 1.",
        )

        return parser
