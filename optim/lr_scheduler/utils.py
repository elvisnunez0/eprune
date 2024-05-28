import math


def linear_schedule(
    t_current: int, t_init: int, t_end: int, init_lr: float, end_lr: float
) -> float:
    """
    Linear schedule for learning rates. The returned learning rate
    will be interpolated between (@t_init, @init_lr) and (@t_end, @end_lr).

    Args:
        t_current: The current epoch or training iteration.
        t_init: The initial epoch or training iteration.
        t_end: The final epoch or training iteration.
        init_lr: The learning rate at step t_init.
        end_lr: The learning rate at step t_end.
    """

    if t_current > t_end:
        lr = end_lr
    else:
        m = (end_lr - init_lr) / (t_end - t_init)
        lr = m * (t_current - t_init) + init_lr

    return lr


def cosine_schedule(
    t_current: int, t_init: int, t_end: int, init_lr: float, end_lr: float
) -> float:
    """
    Cosine schedule for learning rates. The returned learning rate
    follows a cosine decay starting from (@t_init, @init_lr) to
    (@t_end, @end_lr).

    Args:
        t_current: The current epoch or training iteration.
        t_init: The initial epoch or training iteration.
        t_end: The final epoch or training iteration.
        init_lr: The learning rate at step t_init.
        end_lr: The learning rate at step t_end.
    """

    if t_current > t_end:
        lr = end_lr
    else:
        lr = end_lr + 0.5 * (init_lr - end_lr) * (
            1 + math.cos((t_current - t_init) / (t_end - t_init) * math.pi)
        )

    return lr
