from argparse import ArgumentParser
from typing import Callable, List, Optional

from utils.import_modules import recursive_module_import


class Registry:
    """
    This class is used to register classes so that we can recursively call their
    add_arguments function. This is so that we can gather all of the command line
    arguments to set up training/evaluation.
    """

    def __init__(
        self,
        name: str,
        subdirs: List[str],
        name_separator: Optional[str] = ".",
    ) -> None:
        self.registery_name = name
        self.name_separator = name_separator
        self.subdirs = subdirs
        self.registry = {}

    def _import_subdir_modules(self) -> None:
        """
        Recursively imports that modules in the @self.subdirs directory.
        """
        for subdir in self.subdirs:
            recursive_module_import(subdir=subdir)

    def register(self, name: str, class_type: str = "") -> Callable:
        if class_type != "":
            name = f"{class_type}{self.name_separator}{name}"

        def add_to_registry(item):
            if name in self.registry:
                raise ValueError(
                    f"Attempted to register {name} twice in registry {self.registery_name}."
                )
            self.registry[name] = item
            return item

        return add_to_registry

    def get_all_arguments(self, parser: ArgumentParser) -> ArgumentParser:
        """
        Recursively calls the add_arguments function in all of the modules stored in
        @self.registry.

        Args:
            parser: The argument parser used to gather arguments from all modules
                (data, model, optimizer, etc.).
        Returns:
            parser: The original input parser with added groups for the arguments contained
                in this specific registry instance.
        """
        # First import all modules
        self._import_subdir_modules()

        # Iterate through all registered classes and call the add_arguments function.
        for name, module in self.registry.items():
            parser = module.add_arguments(parser)

        return parser

    def __getitem__(self, key: str):
        if key not in self.registry:
            print(
                f"Key {key} not found in {self.registery_name} registry."
                f" Keys in registry are {list(self.registry.keys())}"
            )
        return self.registry[key]
