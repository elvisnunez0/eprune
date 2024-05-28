from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from loguru._logger import Logger
from torch import Tensor
from torch.nn import DataParallel

from layers.ustruct.utils import (get_global_grad2_accs, get_global_params,
                                  magnitude_prune_mask, magnitude_threshold,
                                  rand_mask_like)
from metrics.fisher import Fisher
from models.base_model import BaseModel
from models.pruners.base_pruner import BasePruner


class UStructPruner(BasePruner):
    """
    This is a wrapper class used for pruning models in an unstructured setting.
    """

    def __init__(
        self,
        cfg: Namespace,
        model: BaseModel,
        logger: Logger,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.logger = logger
        self.device = device
        self.enable = getattr(cfg, "model.sparsity.unstructured.enable")
        self.local = getattr(cfg, "model.sparsity.unstructured.local")
        self.target_sparsity = getattr(
            cfg, "model.sparsity.unstructured.target_sparsity"
        )
        self.importance_method = getattr(
            cfg, "model.sparsity.unstructured.importance_method"
        )
        self.reset_path = getattr(cfg, "model.sparsity.reset_path")

        if self.enable and self.target_sparsity is None:
            raise ValueError("Unstructured sparsity target must be specified.")

        # Global pruning (only used if @self.local is False)
        self.global_prune_conv = getattr(
            cfg, "model.sparsity.unstructured.global.prune_conv"
        )
        self.global_prune_linear = getattr(
            cfg, "model.sparsity.unstructured.global.prune_linear"
        )

        if self.importance_method == "magnitude":
            self.magnitude_prune()
        elif self.importance_method == "random":
            self.rand_prune()
        elif self.importance_method == "fisher":
            self.fisher_prune()

    def reset_weights(self) -> None:
        if self.reset_path is not None:
            if isinstance(self.model, DataParallel):
                setattr(self.model.module, "state_dict_path", self.reset_path)
                self.model.module.load_from_state_dict()
            else:
                setattr(self.model, "state_dict_path", self.reset_path)
                self.model.load_from_state_dict()

    @torch.no_grad()
    def magnitude_prune(self) -> None:
        """
        Performs magnitude pruning either on a local or global scale.
        """
        if self.target_sparsity == 0:
            return

        if self.local:
            self.logger.info(f"Applying local magnitude-based unstructured pruning.")
            for module in self.model.modules():
                if hasattr(module, "is_prunable"):
                    if module.is_prunable():
                        prunable_params = module.get_prunable_params()
                        sparsity_scale = getattr(module, "sparsity_scale")
                        target_sparsity = sparsity_scale * self.target_sparsity
                        for param_name, param in prunable_params.items():
                            # Get previous mask
                            mask_suffix = "w" if "weight" in param_name else "b"
                            mask_name = f"mask_{mask_suffix}"
                            prev_mask = None
                            if hasattr(module, mask_name):
                                prev_mask = getattr(module, mask_name)

                            mask = magnitude_prune_mask(
                                tensor=param,
                                target_sparsity=target_sparsity,
                                prev_mask=prev_mask,
                            )
                            module.set_mask(mask, param_name)
        else:  # TODO: Global iterative pruning
            self.logger.info(f"Applying global magnitude-based unstructured pruning.")
            flattened_params = get_global_params(
                model=self.model,
                prune_conv=self.global_prune_conv,
                prune_linear=self.global_prune_linear,
            )
            threshold = magnitude_threshold(
                tensor=flattened_params, target_sparsity=self.target_sparsity
            )

            prunable_modules = []
            if self.global_prune_conv:
                prunable_modules.append(nn.Conv2d)
            if self.global_prune_linear:
                prunable_modules.append(nn.Linear)
            prunable_modules = tuple(prunable_modules)

            for module in self.model.modules():
                if isinstance(module, prunable_modules) and hasattr(
                    module, "is_prunable"
                ):
                    if module.is_prunable():
                        prunable_params = module.get_prunable_params()
                        # sparsity_scale = getattr(module, "sparsity_scale")
                        for param_name, param in prunable_params.items():

                            # Get previous mask (WIP)
                            # mask_suffix = "w" if "weight" in param_name else "b"
                            # mask_name = f"mask_{mask_suffix}"
                            # prev_mask = None
                            # if hasattr(module, mask_name):
                            #     prev_mask = getattr(module, mask_name)

                            mask = magnitude_prune_mask(
                                tensor=param,
                                threshold=threshold,
                                # prev_mask=prev_mask
                            )
                            module.set_mask(mask, param_name)

        # TODO: Currently not being used
        self.reset_weights()

    @torch.no_grad()
    def rand_prune(self) -> None:
        """
        Performs magnitude pruning either on a local or global scale.
        """
        if self.target_sparsity == 0:
            return

        if self.local:
            self.logger.info(f"Applying local random unstructured pruning.")
            for name, module in self.model.named_modules():
                if hasattr(module, "is_prunable"):
                    if module.is_prunable():
                        prunable_params = module.get_prunable_params()
                        sparsity_scale = getattr(module, "sparsity_scale")
                        target_sparsity = sparsity_scale * self.target_sparsity
                        for param_name, param in prunable_params.items():
                            mask = rand_mask_like(
                                param,
                                fraction_zeros=target_sparsity,
                                device=param.device,
                            )
                            module.set_mask(mask, param_name)
        # else:  # TODO: Global iterative pruning
        #     self.logger.info(f"Applying global random unstructured pruning.")
        #     flattened_params = get_global_params(
        #         model=self.model,
        #         prune_conv=self.global_prune_conv,
        #         prune_linear=self.global_prune_linear,
        #     )
        #     rand_params = torch.rand_like(flattened_params)
        #     threshold = magnitude_threshold(
        #         tensor=flattened_params, target_sparsity=self.target_sparsity
        #     )

        #     prunable_modules = []
        #     if self.global_prune_conv:
        #         prunable_modules.append(nn.Conv2d)
        #     if self.global_prune_linear:
        #         prunable_modules.append(nn.Linear)
        #     prunable_modules = tuple(prunable_modules)

        #     for module in self.model.modules():
        #         if isinstance(module, prunable_modules) and hasattr(
        #             module, "is_prunable"
        #         ):
        #             if module.is_prunable():
        #                 prunable_params = module.get_prunable_params()
        #                 # sparsity_scale = getattr(module, "sparsity_scale")
        #                 for param_name, param in prunable_params.items():

        #                     # Get previous mask (WIP)
        #                     # mask_suffix = "w" if "weight" in param_name else "b"
        #                     # mask_name = f"mask_{mask_suffix}"
        #                     # prev_mask = None
        #                     # if hasattr(module, mask_name):
        #                     #     prev_mask = getattr(module, mask_name)

        #                     mask = magnitude_prune_mask(
        #                         tensor=param,
        #                         threshold=threshold,
        #                         # prev_mask=prev_mask
        #                     )
        #                     module.set_mask(mask, param_name)

    @torch.no_grad()
    def fisher_prune(self) -> None:
        """
        Performs pruning based on the trace of the FIM either on a local or global scale.
        """
        if self.target_sparsity == 0:
            return

        # Compute the squared gradient for each parameter according to FIM computation.
        # Populates the grad2_acc attribute of parameters.
        fisher = Fisher(cfg=self.cfg, device=self.device, logger=self.logger)
        self.logger.info(f"Computing Fisher information...")
        fisher.compute_fisher(model=self.model)

        if self.local:
            self.logger.info(f"Applying local Fisher-based unstructured pruning.")
            for name, module in self.model.named_modules():
                if hasattr(module, "is_prunable"):
                    if module.is_prunable():
                        prunable_params = module.get_prunable_params()
                        sparsity_scale = getattr(module, "sparsity_scale")
                        target_sparsity = sparsity_scale * self.target_sparsity
                        for param_name, param in prunable_params.items():
                            # Get previous mask (if doing iterative pruning)
                            mask_suffix = "w" if "weight" in param_name else "b"
                            mask_name = f"mask_{mask_suffix}"
                            prev_mask = None
                            if hasattr(module, mask_name):
                                prev_mask = getattr(module, mask_name)

                            param_grad2_acc = getattr(param, "grad2_acc")
                            mask = magnitude_prune_mask(
                                tensor=param_grad2_acc,
                                target_sparsity=target_sparsity,
                                prev_mask=prev_mask,
                            )

                            module.set_mask(mask, param_name)
        else:
            self.logger.info(f"Applying global Fisher-based unstructured pruning.")
            flattened_grad2_accs = get_global_grad2_accs(
                model=self.model,
                prune_conv=self.global_prune_conv,
                prune_linear=self.global_prune_linear,
            )
            threshold = magnitude_threshold(
                tensor=flattened_grad2_accs, target_sparsity=self.target_sparsity
            )

            prunable_modules = []
            if self.global_prune_conv:
                prunable_modules.append(nn.Conv2d)
            if self.global_prune_linear:
                prunable_modules.append(nn.Linear)
            prunable_modules = tuple(prunable_modules)

            for module in self.model.modules():
                if isinstance(module, prunable_modules) and hasattr(
                    module, "is_prunable"
                ):
                    if module.is_prunable():
                        prunable_params = module.get_prunable_params()
                        # sparsity_scale = getattr(module, "sparsity_scale")
                        for param_name, param in prunable_params.items():

                            # Get previous mask (WIP)
                            # mask_suffix = "w" if "weight" in param_name else "b"
                            # mask_name = f"mask_{mask_suffix}"
                            # prev_mask = None
                            # if hasattr(module, mask_name):
                            #     prev_mask = getattr(module, mask_name)

                            param_grad2_acc = getattr(param, "grad2_acc")
                            mask = magnitude_prune_mask(
                                tensor=param_grad2_acc,
                                threshold=threshold,
                                # prev_mask=prev_mask
                            )
                            module.set_mask(mask, param_name)

        # Delete Fisher-related parameters
        fisher.clear_fisher_params(model=self.model)

    def compute_sparsity_from_mask(self) -> float:
        """
        Iterates through the model's modules and gathers the number of zeros
        in the module masks. Based on this, computes the global sparsity rate.
        """
        total_params = sum([p.numel() for p in self.model.parameters()])
        total_num_zeros = 0

        for module in self.model.modules():
            if hasattr(module, "is_prunable") and module.is_prunable():
                num_zeros, _ = module.count_zeros()
                total_num_zeros += num_zeros

        global_sparsity = total_num_zeros / total_params

        return global_sparsity

    def compute_sparsity(self) -> float:
        num_zeros = 0
        num_elem = 0
        for param in self.model.parameters():
            num_elem += param.numel()
            num_zeros += torch.eq(param, 0).sum()
        return num_zeros / num_elem

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> ArgumentParser:
        if cls != UStructPruner:
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--model.sparsity.unstructured.enable",
            action="store_true",
            help="Whether to prune the model using unstructured pruning. Defaults to False.",
        )

        group.add_argument(
            "--model.sparsity.unstructured.local",
            action="store_true",
            default=True,
            help="Whether to apply the importance selection on a local (True) or global (False) scale. "
            "Defaults to True.",
        )

        # In LTH, the final linear layer is pruned at half the rate of the conv layers
        group.add_argument(
            "--model.sparsity.unstructured.output-sparsity-scale",
            type=float,
            default=1.0,
            help="The sparsity rate of the output layer will be scaled by this value. Defaults to 1.",
        )

        group.add_argument(
            "--model.sparsity.unstructured.target-sparsity",
            type=float,
            default=None,
            help="The target sparsity rate. This is the fraction of zeros in the model. Defaults to None",
        )

        group.add_argument(
            "--model.sparsity.unstructured.importance-method",
            type=str,
            default="magnitude",
            help="The importance selection method. Defaults to magnitude.",
        )

        group.add_argument(
            "--model.sparsity.unstructured.prune-bias",
            action="store_true",
            help="Whether biases should be pruned. Defaults to False.",
        )

        # TODO: This arg should be moved elsewhere
        group.add_argument(
            "--model.sparsity.reset-path",
            type=str,
            default=None,
            help="The path to the model checkpoint that will be used to "
            "initialize weights after pruning. Defaults to None.",
        )

        # TODO: This arg should be moved elsewhere
        group.add_argument(
            "--model.sparsity.unstructured.ckpt-num",
            type=int,
            default=None,
            help="When doing experiments where a checkpoint is loaded "
            "and then pruned, this specifies the checkpoint number.",
        )

        # Iterative pruning
        group.add_argument(
            "--model.sparsity.unstructured.iterative.final-target-sparsity",
            type=float,
            default=None,
            help="The final target sparsity rate for iterative pruning. This is the "
            "fraction of zeros in the model. Defaults to None.",
        )

        group.add_argument(
            "--model.sparsity.unstructured.iterative.iterations",
            type=int,
            default=None,
            help="The number of pruning iterations to attain the final target sparsity. "
            "Defaults to None.",
        )

        group.add_argument(
            "--model.sparsity.unstructured.iterative.final-ckpt-name",
            type=str,
            default=None,
            help="The name of the checkpoint to load at the start of each iteration. "
            "Defaults to None.",
        )

        # Global pruning
        group.add_argument(
            "--model.sparsity.unstructured.global.prune-conv",
            action="store_true",
            default=True,
            help="Whether to prune conv layers. Defaults to True.",
        )
        group.add_argument(
            "--model.sparsity.unstructured.global.prune-linear",
            action="store_true",
            default=True,
            help="Whether to prune linear layers. Defaults to True.",
        )
        return parser
