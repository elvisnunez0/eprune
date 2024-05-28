from argparse import Namespace

from models.base_model import BaseModel


class BasePruner(object):
    def __init__(self, cfg: Namespace, model: BaseModel) -> None:
        self.cfg = cfg
        self.model = model
