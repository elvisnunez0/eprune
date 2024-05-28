from argparse import ArgumentParser, Namespace

from optim.lr_scheduler import LR_SCHEDULER_REGISTRY
from optim.lr_scheduler.base_lr_scheduler import BaseLRScheduler


@LR_SCHEDULER_REGISTRY.register(name="multistep", class_type="lr_scheduler")
class MultiStepLRScheduler(BaseLRScheduler):
    def __init__(self, cfg: Namespace) -> None:
        super().__init__(cfg)

        self.gamma = getattr(cfg, "optim.lr_scheduler.multistep.gamma")
        self.per_epoch = getattr(cfg, "optim.lr_scheduler.multistep.per_epoch")
        self.milestones = getattr(cfg, "optim.lr_scheduler.multistep.milestones")
        self.current_lr = self.max_lr

        self.milestones_count = 0

    def get_lr(self, epoch: int) -> float:
        lr = self.get_warmup_lr(epoch=epoch)

        # In this case, we are not in the warmup phase
        if lr is None:
            if self.per_epoch:
                # epoch - 1 because epoch counts start at 1
                lr = self.max_lr * (self.gamma ** (epoch - 1))
            else:
                if epoch in self.milestones:
                    self.milestones_count += 1
                lr = self.max_lr * (self.gamma**self.milestones_count)

        return lr

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--optim.lr-scheduler.multistep.gamma",
            type=float,
            default=0.1,
            help="The factor to decay the learning rate by at every epoch "
            "if --optim.lr_scheduler.multistep.per_epoch is true. If False, "
            "will decay the learning rate at each milestone.",
        )

        group.add_argument(
            "--optim.lr-scheduler.multistep.per-epoch",
            action="store_true",
            default=True,
            help="Whether to decay the learning rate at every epoch. If False, "
            "will decay at every milestone. Defaults to True.",
        )

        group.add_argument(
            "--optim.lr-scheduler.multistep.milestones",
            type=list,
            default=None,
            help="The epochs at which to decay the learning rate.",
        )

        return parser
