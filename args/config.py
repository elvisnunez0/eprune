import sys
from argparse import ArgumentParser

from args.utils import load_config
from data.datasets import get_dataset_arguments
from data.transforms import get_transform_arguments
from loss import get_loss_arguments
from metrics import get_metric_arguments
from models import get_model_arguments
from optim.lr_scheduler import get_lr_scheduler_arguments
from optim.optimizer import get_optim_arguments
from saving import get_saver_arguments


def get_common_arguments(parser: ArgumentParser) -> ArgumentParser:
    """
    Adds arguments common to all tasks.
    """
    group = parser.add_argument_group("Common arguments")

    group.add_argument(
        "--common.task",
        type=str,
        default="classification",
        help="The type of task used to train/eval. Defaults to classification.",
    )

    group.add_argument(
        "--common.eval-only",
        action="store_true",
        help="If true, will not train the model and only evaluate. Defaults to False.",
    )

    group.add_argument(
        "--common.config",
        type=str,
        help="The path to a yaml config containing the training parameters.",
    )

    group.add_argument(
        "--common.logger.save-name",
        type=str,
        help="The name of the log file to save.",
    )

    group.add_argument(
        "--common.logger.frequency",
        type=int,
        default=100,
        help="The number of iterations between logging updates during training. Defaults to 100.",
    )

    group.add_argument(
        "--common.logger.rotation",
        type=str,
        default="5 MB",
        help="The rotation string specifying when to create a new log. Defaults to 5 MB.",
    )

    group.add_argument(
        "--common.logger.level",
        type=str,
        default="DEBUG",
        help="The minimum severity level for the logger. Defaults to DEBUG.",
    )

    group.add_argument(
        "--common.distributed-training.data-parallel",
        action="store_true",
        default=True,
        help="Whether to train with data parallelism. Defaults to True when available.",
    )

    return parser


def get_arguments(update_from_config: bool = True, manual_overwrite: bool = True):
    """
    Gets all arguments needed for training/eval.

    Args:
        update_from_config: If true, will update the default arguments with the
            values in the config specified in 'common.config'
        manual_overwrite: If true, will replace the default arguments (after
            updating them from the specified config) with arguments specified
            in the command line.
    """
    parser = ArgumentParser(description="Command line arguments", add_help=True)

    # Add common arguments
    parser = get_common_arguments(parser)

    # Add dataset arguments
    parser = get_dataset_arguments(parser)

    # Add transformation arguments
    parser = get_transform_arguments(parser)

    # # Add model arguments
    parser = get_model_arguments(parser)

    # # Add optimizer arguments
    parser = get_optim_arguments(parser)

    # # Add LR scheduler arguments
    parser = get_lr_scheduler_arguments(parser)

    # # Add loss arguments
    parser = get_loss_arguments(parser)

    # # Add saver arguments
    parser = get_saver_arguments(parser)

    # # Add metrics argumetns
    parser = get_metric_arguments(parser)

    args = parser.parse_args()

    cfg_path = getattr(args, "common.config")
    if update_from_config and cfg_path is not None:
        config = load_config(cfg_path, flatten=True)

        # Update args with those in the config
        for key, value in config.items():
            setattr(args, key, value)

    if manual_overwrite:
        argv = sys.argv

        for i, arg in enumerate(sys.argv):
            if "--" in arg:
                # Since flag args are specified with '-', replace with '_'
                # [2:] to skip first '--' in flag
                arg_name = arg[2:].replace("-", "_")
                setattr(args, arg_name, argv[i + 1])

    return args
