from argparse import Namespace
from typing import Optional

from torch.nn import DataParallel

from args.config import get_arguments
from data.loaders import create_loader
from engine.eval import Evaluator
from engine.training import Trainer
from logger import configure_logger, logger
from loss import get_loss_from_registry
from models import get_model_from_registry
from models.pruners.ustruct_pruner import UStructPruner
from models.pruners.utils import update_cfg_ckpt_args
from optim.lr_scheduler import get_lr_scheduler_from_registry
from optim.optimizer import get_optim_from_registry
from utils.device import get_device
from utils.loading import get_start_epoch


def main(cfg: Optional[Namespace] = None):
    if cfg is None:
        cfg = get_arguments(update_from_config=True)

    # Training callbacks
    training_callbacks = []

    # Determine if we are running in eval only mode
    eval_only = getattr(cfg, "common.eval_only")

    # Set logger
    configure_logger(cfg)

    # If training from a checkpoint, update config accordingly
    ckpt_num = getattr(cfg, "model.load_from_state_dict.from_ckpt.ckpt_num")
    if ckpt_num is not None:
        update_cfg_ckpt_args(cfg=cfg, logger=logger)

    # Get model
    device = get_device()
    model = get_model_from_registry(cfg=cfg)

    # Set distributed training
    data_parallel = getattr(cfg, "common.distributed_training.data_parallel")
    model = model.to(device=device)

    if device == "cuda" and data_parallel:
        model = DataParallel(model)

    # Set pruning module
    uprune_enable = getattr(cfg, "model.sparsity.unstructured.enable")
    if uprune_enable:
        pruner = UStructPruner(cfg=cfg, model=model, logger=logger, device=device)

        sparsity = pruner.compute_sparsity_from_mask()
        if pruner.local:
            logger.info(
                f"Applied local pruning. Set unstructured model sparsity to {sparsity:0.4f}."
            )
        else:
            logger.info(
                f"Applied global pruning. Set unstructured model sparsity to {sparsity:0.4f}."
            )

    # Get test loader
    test_loader = create_loader(cfg, mode="test", shuffle=False)

    # Get loss function
    criteria = get_loss_from_registry(cfg=cfg)
    criteria = criteria.to(device)

    if eval_only:
        evaluator = Evaluator(
            cfg=cfg,
            model=model,
            criteria=criteria,
            logger=logger,
            device=device,
            loader=test_loader,
        )

        evaluator.run()
    else:
        # Get train loader
        train_loader = create_loader(cfg, mode="train", shuffle=True)

        # Get loader for train data that does not apply augmentaions
        train_loader_no_aug = create_loader(cfg, mode="train_no_aug", shuffle=False)

        # Get optimizer
        optimizer = get_optim_from_registry(cfg=cfg, model_params=model.parameters())

        # Get start epoch
        start_epoch = get_start_epoch(cfg=cfg, logger=logger)

        # Get learning rate scheduler
        lr_scheduler = get_lr_scheduler_from_registry(cfg=cfg)

        trainer = Trainer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criteria=criteria,
            start_epoch=start_epoch,
            logger=logger,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            train_loader_no_aug=train_loader_no_aug,
            callbacks=training_callbacks,
        )

        trainer.run()


if __name__ == "__main__":
    cfg = get_arguments()
    main(cfg=cfg)
