from argparse import Namespace

from loguru import logger
from loguru._logger import Logger


def configure_logger(cfg: Namespace) -> Logger:
    name = getattr(cfg, "common.logger.name")
    rotation = getattr(cfg, "common.logger.rotation")
    level = getattr(cfg, "common.logger.level")
    logger.add(f"{name}.log", rotation=rotation, level=level)

    return logger
