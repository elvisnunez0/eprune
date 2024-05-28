from argparse import ArgumentParser, Namespace
from functools import partial

from optim.lr_scheduler import LR_SCHEDULER_REGISTRY
from optim.lr_scheduler.base_lr_scheduler import BaseLRScheduler
from optim.lr_scheduler.utils import cosine_schedule


@LR_SCHEDULER_REGISTRY.register(name="cosine", class_type="lr_scheduler")
class CosineLRScheduler(BaseLRScheduler):
    def __init__(self, cfg: Namespace) -> None:
        super().__init__(cfg)

        t_init = self.warmup_epochs if self.warmup_epochs > 0 else 1
        t_end = getattr(cfg, "optim.lr_scheduler.cosine.final_epoch")
        t_end = self.total_epochs if t_end is None else t_end

        self.schedule = partial(
            cosine_schedule,
            t_init=t_init,
            t_end=t_end,
            init_lr=self.max_lr,
            end_lr=self.min_lr,
        )

    def get_lr(self, epoch: int) -> float:
        lr = self.get_warmup_lr(epoch=epoch)

        # In this case, we are not in the warmup phase
        if lr is None:
            if epoch > self.total_epochs:
                epoch = self.total_epochs
            lr = self.schedule(t_current=epoch)

        return lr

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--optim.lr-scheduler.cosine.final_epoch",
            type=int,
            default=None,
            help="The epoch at which the schedule will reach its final value. "
            "Defaults to None.",
        )

        return parser
