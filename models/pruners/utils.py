import os
from argparse import Namespace

from loguru._logger import Logger


def lr_scaler(base_lr: float, ckpt_num: float, epochs: int = 200) -> float:
    """
    Scales the learning rate according to the following equation:
        new lr = base_lr * (1 - 0.9/200 * ckpt_num)

    Args:
        base_lr: The base learning rate. Typically the max learning rate used to
            train the original model.
        ckpt_num: The checkpoint number being loaded.
        epochs: The number of epochs the original model was trained for.
    """
    return base_lr * (1 - 0.9 / 200 * ckpt_num)


def update_cfg_ckpt_args(cfg: Namespace, logger: Logger) -> Namespace:
    ckpt_num = int(getattr(cfg, "model.load_from_state_dict.from_ckpt.ckpt_num"))

    # Update the learning rate.
    base_lr = getattr(cfg, "optim.lr_scheduler.max_lr")
    original_epochs = getattr(
        cfg, "model.load_from_state_dict.from_ckpt.original_training_epochs"
    )
    new_lr = lr_scaler(base_lr=base_lr, ckpt_num=ckpt_num, epochs=original_epochs)
    setattr(cfg, "optim.lr_scheduler.max_lr", new_lr)

    # Set the max number of epochs.
    setattr(cfg, "optim.lr_scheduler.max_epochs", original_epochs - ckpt_num)

    # Update the path of the state dict to load.
    base_sd_dir = getattr(cfg, "model.load_from_state_dict.path")
    sd_path = os.path.join(base_sd_dir, f"ckpt{ckpt_num}.pth")
    setattr(cfg, "model.load_from_state_dict.path", sd_path)

    # Update the save directory.
    base_savedir = getattr(cfg, "saver.dir")
    savedir = os.path.join(base_savedir, f"ckpt{ckpt_num}")
    setattr(cfg, "saver.dir", savedir)

    logger.info(f"Set max learning rate to {getattr(cfg, 'optim.lr_scheduler.max_lr')}")
    logger.info(
        f"Set number of training epochs to {getattr(cfg, 'optim.lr_scheduler.max_epochs')}"
    )
    logger.info(
        f"Set state dict path to {getattr(cfg, 'model.load_from_state_dict.path')}"
    )
    logger.info(f"Set save dir to {getattr(cfg, 'saver.dir')}")
