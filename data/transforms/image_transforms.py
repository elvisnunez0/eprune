from argparse import ArgumentParser, Namespace
from typing import Optional, Sequence

from torchvision import transforms as T

from data.transforms import TRANSFORMS_REGISTRY
from data.transforms.base_transform import BaseTransform
from data.transforms.utils import get_interpolation_mode


@TRANSFORMS_REGISTRY.register(name="random_resized_crop", class_type="image_transform")
class RandomResizedCrop(BaseTransform, T.RandomResizedCrop):
    def __init__(
        self, cfg: Namespace, size: Optional[int] = None, *args, **kwargs
    ) -> None:
        super().__init__(cfg, *args, **kwargs)

        if size is None:
            size = getattr(cfg, "dataset.image_transforms.random_resized_crop.size")

        scale = getattr(cfg, "dataset.image_transforms.random_resized_crop.scale")
        ratio = getattr(cfg, "dataset.image_transforms.random_resized_crop.ratio")
        interpolation = getattr(
            cfg, "dataset.image_transforms.random_resized_crop.interpolation"
        )

        T.RandomResizedCrop.__init__(
            self,
            size=size,
            scale=scale,
            ratio=ratio,
            interpolation=get_interpolation_mode(interpolation),
        )

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--dataset.image-transforms.random-resized-crop.enable",
            action="store_true",
            help="Whether to enable the RandomResizedCrop augmentaion.",
        )

        group.add_argument(
            "--dataset.image-transforms.random-resized-crop.size",
            type=int or tuple,
            help="The size of the crop. Either an int or a tuple.",
        )

        group.add_argument(
            "--dataset.image-transforms.random-resized-crop.scale",
            type=tuple,
            default=(0.08, 1.0),
            help="Lower and upper bounds of the random area to crop. Defaults to (0.08, 1.0).",
        )

        group.add_argument(
            "--dataset.image-transforms.random-resized-crop.ratio",
            type=tuple,
            default=(3.0 / 4.0, 4.0 / 3.0),
            help="Lower and upper bounds of the random aspect ratio before resizing. Defaults to (3/4, 4/3).",
        )

        group.add_argument(
            "--dataset.image-transforms.random-resized-crop.interpolation",
            type=str,
            default="bilinear",
            help="The type of interpolation to use. Defaults to bilinear.",
        )

        return parser


@TRANSFORMS_REGISTRY.register(name="random_crop", class_type="image_transform")
class RandomCrop(BaseTransform, T.RandomCrop):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        size = getattr(cfg, "dataset.image_transforms.random_crop.size")
        padding = getattr(cfg, "dataset.image_transforms.random_crop.padding")

        T.RandomCrop.__init__(self, size=size, padding=padding)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--dataset.image-transforms.random-crop.enable",
            action="store_true",
            help="Whether to enable random cropping.",
        )

        group.add_argument(
            "--dataset.image-transforms.random-crop.size",
            type=int,
            default=None,
            help="The size of the crop. Defaults to None.",
        )

        group.add_argument(
            "--dataset.image-transforms.random-crop.padding",
            type=int,
            default=None,
            help="The padding to use on the image borders. Defaults to None.",
        )

        return parser


@TRANSFORMS_REGISTRY.register(
    name="random_horizontal_flip", class_type="image_transform"
)
class RandomHorizontalFlip(BaseTransform, T.RandomHorizontalFlip):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        p = getattr(cfg, "dataset.image_transforms.random_horizontal_flip.p")

        T.RandomHorizontalFlip.__init__(self, p=p)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--dataset.image-transforms.random-horizontal-flip.enable",
            action="store_true",
            help="Whether to enable random horizontal flipping.",
        )

        group.add_argument(
            "--dataset.image-transforms.random-horizontal-flip.p",
            type=float,
            default=0.5,
            help="The probability of randomly flipping the input sample. Defaults to 0.5.",
        )

        return parser


@TRANSFORMS_REGISTRY.register(name="resize", class_type="image_transform")
class Resize(BaseTransform, T.Resize):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        size = getattr(cfg, "dataset.image_transforms.resize.size")
        interpolation = getattr(cfg, "dataset.image_transforms.resize.interpolation")

        T.Resize.__init__(
            self, size=size, interpolation=get_interpolation_mode(interpolation)
        )

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--dataset.image-transforms.resize.enable",
            action="store_true",
            help="Whether to enable the Resize augmentaion.",
        )

        group.add_argument(
            "--dataset.image-transforms.resize.size",
            type=int or tuple,
            help="The resolution to resize the image to. Either an int or a tuple.",
        )

        group.add_argument(
            "--dataset.image-transforms.resize.interpolation",
            type=str,
            default="bilinear",
            help="The type of interpolation to use for resizing. Defaults to bilinear.",
        )

        return parser


@TRANSFORMS_REGISTRY.register(name="normalize", class_type="image_transform")
class Normalize(BaseTransform, T.Normalize):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        mean = getattr(cfg, "dataset.image_transforms.normalize.mean")
        std = getattr(cfg, "dataset.image_transforms.normalize.std")
        inplace = getattr(cfg, "dataset.image_transforms.normalize.inplace")

        T.Normalize.__init__(self, mean=mean, std=std, inplace=inplace)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--dataset.image-transforms.normalize.enable",
            action="store_true",
            help="Whether to enable the Normalize augmentaion.",
        )

        group.add_argument(
            "--dataset.image-transforms.normalize.mean",
            type=Sequence,
            default=[0.0, 0.0, 0.0],
            help="The mean to use to standardize the image."
            " Defaults to [0.0, 0.0, 0.0]",
        )

        group.add_argument(
            "--dataset.image-transforms.normalize.std",
            type=Sequence,
            default=[1.0, 1.0, 1.0],
            help="The standard deviation to use to standardize the image."
            " Defaults to [1.0, 1.0, 1.0]",
        )

        group.add_argument(
            "--dataset.image-transforms.normalize.inplace",
            action="store_true",
            help="Whether normalization should be performed inplace.",
        )

        return parser


@TRANSFORMS_REGISTRY.register(name="center_crop", class_type="image_transform")
class CenterCrop(BaseTransform, T.CenterCrop):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        size = getattr(cfg, "dataset.image_transforms.center_crop.size")

        T.CenterCrop.__init__(self, size=size)

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--dataset.image-transforms.center-crop.enable",
            action="store_true",
            help="Whether to enable center crop augmentaion.",
        )

        group.add_argument(
            "--dataset.image-transforms.center-crop.size",
            type=int or tuple,
            help="The resolution to resize the image to. Either an int or a tuple.",
        )

        return parser


@TRANSFORMS_REGISTRY.register(name="to_tensor", class_type="image_transform")
class ToTensor(BaseTransform, T.ToTensor):
    def __init__(self, cfg: Namespace, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        T.ToTensor.__init__(self)
