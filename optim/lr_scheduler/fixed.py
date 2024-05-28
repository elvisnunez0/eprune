from argparse import ArgumentParser, Namespace

from optim.lr_scheduler import LR_SCHEDULER_REGISTRY
from optim.lr_scheduler.base_lr_scheduler import BaseLRScheduler


@LR_SCHEDULER_REGISTRY.register(name="fixed", class_type="lr_scheduler")
class FixedLRScheduler(BaseLRScheduler):
    def __init__(self, cfg: Namespace) -> None:
        super().__init__(cfg)

    def get_lr(self, epoch: int) -> float:
        return self.max_lr

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        return parser
