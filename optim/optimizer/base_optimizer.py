from argparse import ArgumentParser, Namespace

from optim.optimizer import OPTIM_REGISTRY


@OPTIM_REGISTRY.register(name="__base__", class_type="optimizer")
class BaseOptimizer(object):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        self.weight_decay = getattr(cfg, "optim.weight_decay")

        self.load_model_state_dict = getattr(
            cfg, "optim.load_from_state_dict.load_from_model_state_dict"
        )
        self.state_dict_path = getattr(cfg, "optim.load_from_state_dict.path")
        self.state_dict_key = getattr(cfg, "optim.load_from_state_dict.key")

        if self.load_model_state_dict:
            self.state_dict_path = getattr(cfg, "model.load_from_state_dict.path")

        # self.state_dict_strict = getattr(cfg, "optim.load_from_state_dict.strict")

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--optim.name",
            type=str,
            default="sgd",
            help="Which optimizer to use. Defaults to SGD.",
        )

        group.add_argument(
            "--optim.weight-decay",
            type=float,
            default=4e-5,
            help="Weight decay strength. Defaults to 4e-5.",
        )

        group.add_argument(
            "--optim.load-from-state-dict.path",
            type=str,
            default=None,
            help="The path to a dictionary containing the optimizer's state dictionary."
            "Defaults to None. If the model state dict is provided, wilil use that path.",
        )

        group.add_argument(
            "--optim.load-from-state-dict.strict",
            action="store_true",
            default=True,
            help="Whether loading from the state dict should be strict. Defaults to True.",
        )

        group.add_argument(
            "--optim.load-from-state-dict.key",
            type=str,
            default="optimizer",
            help="The key in the provided state dict path that contains the optimizer state dict."
            "Defaults to 'optimizer'.",
        )

        group.add_argument(
            "--optim.load-from-state-dict.load-from-model-state-dict",
            action="store_true",
            default=True,
            help="If set to true, and the model state dict is provided, will attempt to load the"
            "optimizer state dict from there. Defaults to True.",
        )

        return parser
