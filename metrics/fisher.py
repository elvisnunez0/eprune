from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru._logger import Logger
from tqdm import tqdm

from metrics import METRIC_REGISTRY
from metrics.post_train_metric import PostTrainMetric


@METRIC_REGISTRY.register("fisher", class_type="metric")
class Fisher(PostTrainMetric):
    def __init__(self, cfg: Namespace, device: torch.device, logger: Logger) -> None:
        super().__init__(cfg=cfg, device=device, logger=logger)

        self.method = getattr(cfg, "metrics.post_train.fisher.method")
        self.mc_epochs = getattr(
            cfg,
            "metrics.post_train.fisher.mc_epochs",
        )

        self.per_layer_traces = None

    def _per_layer_fisher(self, model: nn.Module) -> OrderedDict:
        """
        Computes the Fisher trace on a per-layer basis. Assumes that the
        squared gradients have already been set in the parameters via MC.
        """
        trace_dict = OrderedDict()
        layers = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)

        for name, module in model.named_modules():
            if isinstance(module, layers):
                weight_trace = module.weight.grad2_acc.cpu().detach().sum().item()
                norm_weight_trace = module.weight.grad2_acc.cpu().detach().mean().item()
                trace_dict[name] = {
                    "weight": weight_trace,
                    "norm_weight": norm_weight_trace,
                }

                if module.bias is not None:
                    bias_trace = module.bias.grad2_acc.cpu().detach().sum().item()
                    norm_bias_trace = module.bias.grad2_acc.mean().item()
                    trace_dict[name]["bias"] = bias_trace
                    trace_dict[name]["norm_bias"] = norm_bias_trace

        return trace_dict

    def compute_fisher(self, model: nn.Module):
        # Prepare model
        if self.recalibrate_bn:
            self.recalibrate_model(model, self.data_loader)

        if self.eval_mode:
            model.eval()

        for p in model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0

        with torch.enable_grad():
            for k in range(self.mc_epochs):
                for i, batch in enumerate(
                    tqdm(self.data_loader, leave=False, desc="Computing Fisher")
                ):
                    data = batch["sample"].to(self.device)
                    output = model(data)
                    # The gradients used to compute the FIM needs to be for y sampled from
                    # the model distribution y ~ p_w(y|x), not for y from the dataset.
                    target = (
                        torch.multinomial(F.softmax(output, dim=-1), 1)
                        .detach()
                        .view(-1)
                    )
                    loss = self.loss_fn(output, target)
                    model.zero_grad()
                    loss.backward()

                    B = data.shape[0]
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad2_acc += B * p.grad.data**2
                            p.grad_counter += 1

        for p in model.parameters():
            if p.grad_counter == 0:
                del p.grad2_acc
            else:
                p.grad2_acc /= p.grad_counter

    def compute_per_layer_fisher(self, model: nn.Module) -> OrderedDict:
        self.compute_fisher(model=model)
        return self._per_layer_fisher(model)

    def clear_fisher_params(self, model: nn.Module):
        for p in model.parameters():
            if hasattr(p, "grad2_acc"):
                del p.grad2_acc
            if hasattr(p, "grad_counter"):
                del p.grad_counter

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--metrics.fisher.method",
            type=str,
            default="monte_carlo",
            help="The method used to estimte the FIM. Defaults to monte_carlo.",
        )

        group.add_argument(
            "--metrics.fisher.mc-epochs",
            type=int,
            default=1,
            help="The number of epochs to use in the MC sampling. Defaults to 1.",
        )

        return parser
