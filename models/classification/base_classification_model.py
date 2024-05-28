from argparse import ArgumentParser, Namespace
from functools import partial

from layers.conv import Conv2d
from layers.linear import Linear
from layers.ustruct.conv import UStructConv2d
from layers.ustruct.linear import UStructLinear
from models import MODEL_REGISTRY
from models.base_model import BaseModel


@MODEL_REGISTRY.register(name="__base__", class_type="classification")
class BaseClassificationModel(BaseModel):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg=cfg, *args, **kwargs)
        self.cfg = cfg

        self.num_classes = getattr(cfg, "model.classification.num_classes")
        self.dropout = getattr(cfg, "model.classification.dropout")

        self.set_sparse_params()

    def set_conv_module(self):
        if self.sparse_model:
            prune_bias = getattr(self.cfg, "model.sparsity.unstructured.prune_bias")
            self.conv_module = partial(UStructConv2d, prune_bias=prune_bias)
        else:
            self.conv_module = Conv2d

    def set_linear_module(self):
        if self.sparse_model:
            prune_bias = getattr(self.cfg, "model.sparsity.unstructured.prune_bias")
            self.linear_module = partial(UStructLinear, prune_bias=prune_bias)
        else:
            self.linear_module = Linear

    def set_sparse_params(self):
        # Set pruning params
        self.sparsity_scale = getattr(
            self.cfg, "model.sparsity.unstructured.output_sparsity_scale"
        )
        self.sparse_model = getattr(self.cfg, "model.sparsity.unstructured.enable")

        # Set conv and linear modules
        self.set_conv_module()
        self.set_linear_module()

        # Sets a global sparse mask. This is used in iterative pruning
        # to keep track of previous masks.
        # self.register_buffer("global_mask", None)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(cls.__name__)

        if cls != BaseClassificationModel:
            return parser

        group.add_argument(
            "--model.classification.name",
            type=str,
            default=None,
            help="The model name to use for classification.",
        )

        group.add_argument(
            "--model.classification.num-classes",
            type=int,
            default=None,
            help="The number of classes in the classification dataset.",
        )

        group.add_argument(
            "--model.classification.dropout",
            action="store_true",
            help="Whether to apply dropout if implemented for the model.",
        )

        return parser
