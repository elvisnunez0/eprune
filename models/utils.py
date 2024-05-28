from typing import List

import torch.nn as nn


def get_maskable_modules(model: nn.Module) -> List[nn.Module]:
    """
    Returns a list of low-level modules that are maskable.
    """
    modules = []

    for module in model.children():
        if hasattr(module, "mask_grad"):
            modules.append(module)
        else:
            modules.extend(get_maskable_modules(module))

    return modules
