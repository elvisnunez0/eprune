from argparse import ArgumentParser, Namespace
from functools import partial

from optim.lr_scheduler.utils import linear_schedule
from optim.optimizer.base_optimizer import BaseOptimizer


class BaseLRScheduler(object):
    def __init__(self, cfg: Namespace) -> None:
        super().__init__()

        self.min_lr = getattr(cfg, "optim.lr_scheduler.min_lr")
        self.max_lr = getattr(cfg, "optim.lr_scheduler.max_lr")
        self.total_epochs = getattr(cfg, "optim.lr_scheduler.max_epochs")
        self.warmup_epochs = getattr(cfg, "optim.lr_scheduler.warmup.epochs")
        self.warmup_initial_lr = getattr(cfg, "optim.lr_scheduler.warmup.initial_lr")

        # NOTE: Epoch counts start at 1, i.e., the first training epoch is epoch 1
        # (as opposed to typically starting at 0).
        self.warmup_lr_fn = partial(
            linear_schedule,
            t_init=1,
            t_end=self.warmup_epochs,
            init_lr=self.warmup_initial_lr,
            end_lr=self.max_lr,
        )

    def get_warmup_lr(self, epoch: int) -> float:
        """
        If @epoch is in the warmup phase, then returns the learning rate at this point.
        The learning rate during the warmup phase is a linear schedule that increases
        linearly from @self.warmup_initial_lr to @self.max_lr over the first @self.warmup_epochs
        epochs. If @epoch is not in the warmup phase, will return None.
        """
        if 1 <= epoch <= self.warmup_epochs:
            return self.warmup_lr_fn(t_current=epoch)
        else:
            return None

    def get_lr(self, epoch: int) -> float:
        """
        Compute the LR at the current epoch.
        """
        raise NotImplementedError()

    def update_lr(self, optimizer: BaseOptimizer, epoch: int) -> BaseOptimizer:
        # Ensure LR is not accidentally set to a negative value.
        lr = max(0.0, self.get_lr(epoch=epoch))

        # TODO: Add LR multipliers here (will be useful for ViT)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return optimizer

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        # If we have a subclass without an add_arguments function, this one will
        # be called again. In that case, we don't want to re-add these arguments.
        if cls != BaseLRScheduler:
            return parser

        group.add_argument(
            "--optim.lr-scheduler.name",
            type=str,
            default="cosine",
            help="The learning rate scheduler. Defaults to cosine.",
        )

        group.add_argument(
            "--optim.lr-scheduler.max-epochs",
            type=int,
            default=None,
            help="The total number of epochs to train for. Defaults to None (must be specified).",
        )

        group.add_argument(
            "--optim.lr-scheduler.max-lr",
            type=float,
            default=None,
            help="The maximum learning rate. Defaults to None (must be specified).",
        )

        group.add_argument(
            "--optim.lr-scheduler.min-lr",
            type=float,
            default=None,
            help="The minimum learning rate to decay to. Defaults to None.",
        )

        group.add_argument(
            "--optim.lr-scheduler.warmup.epochs",
            type=int,
            default=0,
            help="The number of warmup epochs. Defaults to 0.",
        )

        group.add_argument(
            "--optim.lr-scheduler.warmup.initial-lr",
            type=float,
            default=None,
            help="The learning rate at the beginning of the warmup period. Defaults to None.",
        )

        return parser
