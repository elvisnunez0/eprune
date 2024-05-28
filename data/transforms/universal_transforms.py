from argparse import Namespace
from typing import Any, Dict, List

from data.transforms import TRANSFORMS_REGISTRY
from data.transforms.base_transform import BaseTransform


@TRANSFORMS_REGISTRY.register(name="compose", class_type="universal_transform")
class Compose(BaseTransform):
    def __init__(self, cfg: Namespace, transforms_list: List, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.transforms_list = transforms_list

    def __call__(self, data: Dict, key: Any) -> Dict:
        """
        Applies each transformation in self.transforms_list to the data sample.

        Args:
            data: A dict of a data sample. This can contain the image sample itself,
                batch index, etc.
            key: The key name that contains the actual sample to apply the transformation to
                in @data.
        """
        for transform in self.transforms_list:
            data[key] = transform(data[key])
        return data

    def __repr__(self) -> str:
        transforms_str = ",\n  ".join(
            str(transform) for transform in self.transforms_list
        )
        repr_str = f"{self.__class__.__name__}(\n  {transforms_str}\n)"
        return repr_str
