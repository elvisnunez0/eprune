from argparse import ArgumentParser, Namespace

import torch
from torch import nn

from logger import logger
from utils.device import get_device
from utils.loading import load_from_torch


class BaseModel(nn.Module):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.device = get_device()

        self.state_dict_path = getattr(cfg, "model.load_from_state_dict.path")
        self.state_dict_key = getattr(cfg, "model.load_from_state_dict.key")
        self.state_dict_strict = getattr(cfg, "model.load_from_state_dict.strict")
        self.seed = getattr(cfg, "model.seed")

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if self.device != "cpu":
                torch.cuda.manual_seed(self.seed)
            logger.info(f"Set torch seed to {self.seed}.")

    def load_from_state_dict(self):
        if self.state_dict_path is not None:
            state_dict = load_from_torch(self.state_dict_path)

            # If state dict was saved as nn.Dataparallel, remove 'module' prefix
            state_dict = state_dict[self.state_dict_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict=state_dict, strict=self.state_dict_strict)
            logger.info(f"Loaded model state dict from {self.state_dict_path}")

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        if cls != BaseModel:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--model.load-from-state-dict.path",
            type=str,
            default=None,
            help="The path to a dictionary containing the model's state dictionary."
            "Defaults to None.",
        )

        group.add_argument(
            "--model.load-from-state-dict.strict",
            action="store_true",
            default=True,
            help="Whether loading from the state dict should be strict. Defaults to True.",
        )

        group.add_argument(
            "--model.load-from-state-dict.key",
            type=str,
            default="model",
            help="The key in the provided state dict path that contains the model state dict."
            "Defaults to 'model'.",
        )

        group.add_argument(
            "--model.load-from-state-dict.resume-training",
            action="store_true",
            default=True,
            help="Whether training should continue from the checkpoint epoch."
            "Defaults to True.",
        )

        # Args for when loading from a checkpoint.
        group.add_argument(
            "--model.load-from-state-dict.from-ckpt.ckpt-num",
            type=int,
            default=None,
            help="If loading from a directory with multiple files of the form <ckpt_num>.pth, "
            "can specify the number here. Defaults to None.",
        )

        group.add_argument(
            "--model.load-from-state-dict.from-ckpt.original-epochs",
            type=int,
            default=None,
            help="When loading a checkpoint, this specified the number of epochs that the "
            "original model was trained for. Defaults to None.",
        )

        group.add_argument(
            "--model.seed",
            type=int,
            default=None,
            help="The random seed for the model. Defaults to 0.",
        )

        return parser
