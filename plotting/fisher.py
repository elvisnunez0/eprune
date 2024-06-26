import os
import re
from typing import List

import torch
import torch.nn as nn

from args.config import get_arguments
from args.utils import load_config
from models import get_model_from_registry


def get_model_from_config(cfg_path: str) -> nn.Module:
    cfg = get_arguments()
    cfg_dict = load_config(cfg_path=cfg_path, flatten=True)
    for k, v in cfg_dict.items():
        if "model." in k:
            setattr(cfg, k, v)

    # Don't load
    setattr(cfg, "model.load_from_state_dict.path", None)

    model = get_model_from_registry(cfg=cfg)

    return model


def get_module_by_name(model: nn.Module, module_name: str):
    """
    Given the name of a model's module, returns it/
    """
    pattern = re.compile(r"\bmodule\.")
    module_name = pattern.sub("", module_name)  # remove 'module.' from module name

    for model_module_name, module in model.named_modules():
        if model_module_name == module_name:
            return module


def is_type(module: nn.Module, types: List[nn.Module]) -> bool:
    """
    Determines if the type of @module is one of the types in @types.
    """
    for t in types:
        if isinstance(module, t):
            return True

    return False


def get_tr_fisher(fisher_path: str, layers: List[nn.Module] = [nn.Conv2d]):
    """
    Given the path to a directory containing two files:
        - config.yaml (The config file used to compute tr(F))
        - metrics.pth (A dictionary of tr(F) for each training epoch)

    returns the tr(F) values stored in metrics.pth.
    """
    cfg_path = os.path.join(fisher_path, "config.yaml")
    metrics_path = os.path.join(fisher_path, "metrics.pth")

    # A dict of the form metrics[i] = {layer name: }
    metrics = torch.load(metrics_path, map_location="cpu")

    # Get the model to be able to select which layers to consider.
    model = get_model_from_config(cfg_path)

    def get_epoch_stats(epoch: int) -> int:
        epoch_metrics = metrics[epoch]
        tr_fisher_sum = 0

        for module_name, metric_vals in epoch_metrics.items():
            module = get_module_by_name(model, module_name)
            if is_type(module, layers):
                tr_fisher_sum += metric_vals["weight"]
        return tr_fisher_sum

    tr_fisher_per_epoch = [get_epoch_stats(epoch) for epoch in range(len(metrics))]

    return tr_fisher_per_epoch
