from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial
import re
import math
from typing import Literal, Optional, Dict, Any
from collections import OrderedDict

### changed for torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
###

from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
    assert_check_literals,
)


# ============================================================================
# Scheduler Interfaces (same config structure as JAX version)
# ============================================================================


@register_interface
class ScheduleInterface(RegistrableConfigInterface):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Optional[LRScheduler]:
        """Create PyTorch LR scheduler."""
        pass

    @abstractmethod
    def get_lr_at_step(self, step: int) -> float:
        """Get learning rate at specific step (for logging)."""
        pass


@dataclass
class BaseScheduleConfig(ConfigInterface):
    init_value: float = 1e-3
    end_value: float = 1e-3
    steps: int = -1

    def __post_init__(self):
        assert self.steps >= 0


@dataclass
class ConstantScheduleConfig(BaseScheduleConfig):
    end_value: None | float = None
    steps: int = -1

    def __post_init__(self):
        if self.end_value is None:
            self.end_value = self.init_value
        super().__post_init__()
        assert np.allclose(self.init_value, self.end_value)


@register
class ConstantSchedule(ScheduleInterface):
    config: ConstantScheduleConfig

    def __init__(self, config: ConstantScheduleConfig):
        self.config = config

    def create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Optional[LRScheduler]:
        """Constant schedule doesn't need a scheduler."""
        return None

    def get_lr_at_step(self, step: int) -> float:
        return self.config.init_value


@dataclass
class CosineScheduleConfig(BaseScheduleConfig):
    init_value: float = 1e-3
    end_value: float | None = None
    decay_factor: float = 0.1
    steps: int = -1
    exponent: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.end_value is None:
            self.end_value = self.decay_factor * self.init_value
        assert np.allclose(self.end_value, self.decay_factor * self.init_value)


@register
class CosineSchedule(ScheduleInterface):
    config: CosineScheduleConfig

    def __init__(self, config: CosineScheduleConfig):
        self.config = config

    def create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Optional[LRScheduler]:
        """Create cosine annealing scheduler."""
        ### changed for torch
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.steps, eta_min=self.config.end_value)
        ###

    def get_lr_at_step(self, step: int) -> float:
        """Calculate cosine decay learning rate."""
        if step >= self.config.steps:
            return self.config.end_value

        progress = step / self.config.steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.config.end_value + (self.config.init_value - self.config.end_value) * cosine_decay


@dataclass
class WarmupCosineDecayScheduleConfig(BaseScheduleConfig):
    init_value: float = 0.0
    peak_value: float = 1e-3
    decay_factor: float = 1e-1
    warmup_steps: int = -1
    decay_steps: int = -1
    exponent: float = 1.0
    end_value: float | None = None

    def __post_init__(self):
        if self.steps < 0:
            self.steps = self.warmup_steps + self.decay_steps
        super().__post_init__()
        if self.end_value is None:
            self.end_value = self.decay_factor * self.peak_value
        assert np.allclose(self.peak_value * self.decay_factor, self.end_value)
        assert self.warmup_steps > 0
        assert self.decay_steps > 0
        assert self.warmup_steps + self.decay_steps == self.steps


class WarmupCosineAnnealingLR(LRScheduler):
    """Custom PyTorch scheduler for warmup + cosine decay."""

    def __init__(self, optimizer, warmup_steps, decay_steps, init_value, peak_value, end_value, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.init_value = init_value
        self.peak_value = peak_value
        self.end_value = end_value
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            lr = self.init_value + (self.peak_value - self.init_value) * step / self.warmup_steps
        else:
            # Cosine decay
            decay_step = step - self.warmup_steps
            progress = decay_step / self.decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.end_value + (self.peak_value - self.end_value) * cosine_decay

        return [lr for _ in self.optimizer.param_groups]


@register
class WarmupCosineDecaySchedule(ScheduleInterface):
    config: WarmupCosineDecayScheduleConfig

    def __init__(self, config: WarmupCosineDecayScheduleConfig):
        self.config = config

    def create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int) -> Optional[LRScheduler]:
        """Create warmup + cosine decay scheduler."""
        ### changed for torch
        return WarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.decay_steps,
            init_value=self.config.init_value,
            peak_value=self.config.peak_value,
            end_value=self.config.end_value,
        )
        ###

    def get_lr_at_step(self, step: int) -> float:
        """Calculate warmup + cosine decay learning rate."""
        if step < self.config.warmup_steps:
            # Linear warmup
            return (
                self.config.init_value
                + (self.config.peak_value - self.config.init_value) * step / self.config.warmup_steps
            )
        else:
            # Cosine decay
            decay_step = step - self.config.warmup_steps
            if decay_step >= self.config.decay_steps:
                return self.config.end_value
            progress = decay_step / self.config.decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.config.end_value + (self.config.peak_value - self.config.end_value) * cosine_decay


# ============================================================================
# Transform Interfaces (same config structure as JAX version)
# ============================================================================


@dataclass
class GradientTransformConfig(ConfigInterface):
    before_optimizer: bool = True


@register_interface
class GradientTransformInterface(RegistrableConfigInterface):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def apply_transform(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Apply gradient transform (e.g., clipping) after backward pass."""
        pass


@dataclass
class WeightDecayConfig(GradientTransformConfig):
    value: float = 0.0
    mode: Literal["whitelist", "blacklist"] = "whitelist"
    parameter_regex_include: str | None = "((.*weight$)|(.*kernel$))"
    parameter_regex_exclude: str | None = ""

    def __post_init__(self):
        assert_check_literals(self)


@dataclass
class GradClipNormConfig(GradientTransformConfig):
    max_norm: float = 1e8


@dataclass
class GradClipValueConfig(GradientTransformConfig):
    max_delta: float = 1e8


@register
class WeightDecay(GradientTransformInterface):
    config: WeightDecayConfig

    def __init__(self, config: WeightDecayConfig):
        self.config = config

    def get_weight_decay_mask(self, model: nn.Module) -> Dict[str, bool]:
        """Create mask for which parameters should have weight decay."""
        mask = {}
        for name, param in model.named_parameters():
            if self.config.mode == "whitelist":
                if re.match(self.config.parameter_regex_include, name):
                    if not self.config.parameter_regex_exclude or not re.match(
                        self.config.parameter_regex_exclude, name
                    ):
                        mask[name] = True
                    else:
                        mask[name] = False
                else:
                    mask[name] = False
            elif self.config.mode == "blacklist":
                if self.config.parameter_regex_exclude and re.match(self.config.parameter_regex_exclude, name):
                    mask[name] = False
                else:
                    mask[name] = True
            else:
                mask[name] = True
        return mask

    def create_parameter_groups(self, model: nn.Module, base_lr: float) -> list[Dict[str, Any]]:
        """Create parameter groups with different weight decay values."""
        ### changed for torch
        mask = self.get_weight_decay_mask(model)

        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if mask.get(name, False):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": self.config.value, "lr": base_lr})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr})

        return param_groups
        ###

    def apply_transform(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Apply L2 weight decay as additional transform if used as transform."""
        ### changed for torch
        if self.config.value > 0:
            mask = self.get_weight_decay_mask(model)
            for name, param in model.named_parameters():
                if mask.get(name, False) and param.grad is not None:
                    # Add L2 penalty to gradients
                    param.grad.data.add_(param.data, alpha=self.config.value)
        ###


@register
class GradClipNorm(GradientTransformInterface):
    config: GradClipNormConfig

    def __init__(self, config: GradClipNormConfig):
        self.config = config

    def apply_transform(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Apply gradient clipping by global norm."""
        ### changed for torch
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_norm)
        ###


@register
class GradClipValue(GradientTransformInterface):
    config: GradClipValueConfig

    def __init__(self, config: GradClipValueConfig):
        self.config = config

    def apply_transform(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Apply gradient clipping by value."""
        ### changed for torch
        torch.nn.utils.clip_grad_value_(model.parameters(), self.config.max_delta)
        ###


# ============================================================================
# Optimizer Interfaces (same config structure as JAX version)
# ============================================================================


@register_interface
class OptimizerInterface(RegistrableConfigInterface):
    @abstractmethod
    def create_optimizer_and_scheduler(
        self, model: nn.Module, total_steps: int
    ) -> tuple[torch.optim.Optimizer, Optional[LRScheduler], list[GradientTransformInterface]]:
        """Create PyTorch optimizer, scheduler, and gradient transforms."""
        pass


@dataclass
class BaseOptimizerConfig(ConfigInterface):
    learning_rate: float | ScheduleInterface.cfgtype = 1e-3
    transforms: dict[str, GradientTransformInterface.cfgtype] = field(default_factory=dict)
    log_learning_rate: bool = True


@dataclass
class SGDConfig(BaseOptimizerConfig):
    learning_rate: float | ScheduleInterface.cfgtype = 1e-3
    transforms: dict[str, GradientTransformInterface.cfgtype] = field(default_factory=dict)
    momentum: float | None = None
    nesterov: bool = False


@register
class SGD(OptimizerInterface):
    config: SGDConfig

    def __init__(self, config: SGDConfig):
        self.config = config

    def create_optimizer_and_scheduler(
        self, model: nn.Module, total_steps: int
    ) -> tuple[torch.optim.Optimizer, Optional[LRScheduler], list[GradientTransformInterface]]:
        """Create SGD optimizer with scheduler and transforms."""
        # Get learning rate
        if isinstance(self.config.learning_rate, float):
            lr = self.config.learning_rate
            scheduler = None
        else:
            schedule = self.config.learning_rate.instantiate(ScheduleInterface)
            lr = schedule.config.init_value
            scheduler = schedule.create_scheduler(None, total_steps)  # Will set optimizer later

        # Create optimizer
        ### changed for torch
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self.config.momentum or 0.0,
            nesterov=self.config.nesterov,
        )
        ###

        # Update scheduler with optimizer
        if scheduler is not None:
            scheduler.optimizer = optimizer

        # Create transforms
        transforms = [
            transform.instantiate(GradientTransformInterface) for _, transform in self.config.transforms.items()
        ]

        return optimizer, scheduler, transforms


@dataclass
class AdamWConfig(BaseOptimizerConfig):
    learning_rate: float | ScheduleInterface.cfgtype = 1e-3
    transforms: dict[str, GradientTransformInterface.cfgtype] = field(default_factory=dict)
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: WeightDecayConfig = field(default_factory=WeightDecayConfig)
    nesterov: bool = False


@register
class AdamW(OptimizerInterface):
    config: AdamWConfig

    def __init__(self, config: AdamWConfig):
        self.config = config

    def create_optimizer_and_scheduler(
        self, model: nn.Module, total_steps: int
    ) -> tuple[torch.optim.Optimizer, Optional[LRScheduler], list[GradientTransformInterface]]:
        """Create AdamW optimizer with scheduler and transforms."""
        # Get learning rate
        if isinstance(self.config.learning_rate, float):
            lr = self.config.learning_rate
        else:
            lr = self.config.learning_rate.init_value

        # Handle weight decay and create parameter groups
        ### changed for torch
        # Always create weight decay transform to get proper parameter groups
        if isinstance(self.config.weight_decay, float):
            # Create a simple weight decay config
            weight_decay_config = WeightDecayConfig(value=self.config.weight_decay)
            weight_decay_transform = WeightDecay(weight_decay_config)
        else:
            weight_decay_transform = WeightDecay(self.config.weight_decay)

        # Create parameter groups with proper weight decay handling
        param_groups = weight_decay_transform.create_parameter_groups(model, lr)

        # Create optimizer with parameter groups
        optimizer = optim.AdamW(
            param_groups,
            betas=(self.config.b1, self.config.b2),
            eps=self.config.eps,
        )
        ###

        # Update scheduler with optimizer
        if isinstance(self.config.learning_rate, float):
            scheduler = None
        else:
            schedule = self.config.learning_rate.instantiate(ScheduleInterface)
            scheduler = schedule.create_scheduler(optimizer, total_steps)

        # Create transforms
        transforms = [
            transform.instantiate(GradientTransformInterface) for _, transform in self.config.transforms.items()
        ]

        return optimizer, scheduler, transforms


@dataclass
class LambConfig(BaseOptimizerConfig):
    learning_rate: float | ScheduleInterface.cfgtype = 1e-3
    transforms: dict[str, GradientTransformInterface.cfgtype] = field(default_factory=dict)
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-6
    eps_root: float = 0.0
    weight_decay: WeightDecayConfig = field(default_factory=WeightDecayConfig)


@register
class Lamb(OptimizerInterface):
    config: LambConfig

    def __init__(self, config: LambConfig):
        self.config = config

    def create_optimizer_and_scheduler(
        self, model: nn.Module, total_steps: int
    ) -> tuple[torch.optim.Optimizer, Optional[LRScheduler], list[GradientTransformInterface]]:
        """Create LAMB optimizer (using AdamW as placeholder) with scheduler and transforms."""
        # Get learning rate
        if isinstance(self.config.learning_rate, float):
            lr = self.config.learning_rate
            scheduler = None
        else:
            schedule = self.config.learning_rate.instantiate(ScheduleInterface)
            lr = schedule.config.init_value
            scheduler = schedule.create_scheduler(None, total_steps)

        # Handle weight decay and create parameter groups
        ### changed for torch
        # Always create weight decay transform to get proper parameter groups
        if isinstance(self.config.weight_decay, float):
            # Create a simple weight decay config
            weight_decay_config = WeightDecayConfig(value=self.config.weight_decay)
            weight_decay_transform = WeightDecay(weight_decay_config)
        else:
            weight_decay_transform = WeightDecay(self.config.weight_decay)

        # Create parameter groups with proper weight decay handling
        param_groups = weight_decay_transform.create_parameter_groups(model, lr)

        # Create optimizer with parameter groups (using AdamW as LAMB placeholder)
        # TODO: Implement proper LAMB optimizer or use a third-party library
        optimizer = optim.AdamW(
            param_groups,
            betas=(self.config.b1, self.config.b2),
            eps=self.config.eps,
        )
        ###

        # Update scheduler with optimizer
        if scheduler is not None:
            scheduler.optimizer = optimizer

        # Create transforms
        transforms = [
            transform.instantiate(GradientTransformInterface) for _, transform in self.config.transforms.items()
        ]

        return optimizer, scheduler, transforms
