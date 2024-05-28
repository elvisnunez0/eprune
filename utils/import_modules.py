import importlib
import os
from pathlib import Path

# This file is stored in ~/.../cv-library/utils/import_modules.py.
# Set the root directory to the cv-library directory, e.g.,
# ~/.../cv-library.
ROOT_DIR = Path(__file__).parent.parent


def recursive_module_import(subdir: str) -> None:
    """
    This function is used to recursively import modules so that we can gather their
    arguments.

    Args:
        subdir: The (sub) folder in @ROOT_DIR to recursively look through.
            Will search for .py files contained in the directory ROOT_DIR/@subdir.
    """
    for path in ROOT_DIR.glob(os.path.join(".", subdir, "**/*.py")):
        python_filename = path.name

        # Ignore files that start with "_" such as __init__.py files.
        if python_filename[0] != "_":
            # Get the portion of the path that corresponds to ** in glob.
            # E.g., ./loss/classification/cross_entropy.py -> classification
            relative_path = path.relative_to(
                os.path.join(ROOT_DIR, subdir)
            ).with_suffix("")

            # Get the full relative path of the file without .py suffix.
            relative_path = str(Path(os.path.join(subdir, relative_path)))

            # Replace "/" with . for importing syntax
            module_name = relative_path.replace(os.sep, ".")
            importlib.import_module(module_name)
