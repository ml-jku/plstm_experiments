import argparse
from functools import partial
from typing import Callable, Dict, Any, Tuple
from dataclasses import dataclass

### changed for torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.amp import autocast, GradScaler
from plstm.torch.dtype import str_dtype_to_torch

###
import numpy as np
from chex import Array, ArrayTree

from .simple_dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .utils import AverageMeter

### changed for torch
from .optimizers_torch import OptimizerInterface, GradientTransformInterface

###
from .model_wrappers_torch import BaseModelTorch

from dataclasses import dataclass, field
from compoconf import ConfigInterface, register_interface, register, RegistrableConfigInterface


### changed for torch
@dataclass
class TorchTrainState:
    """PyTorch training state."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: LRScheduler | None
    transforms: list[GradientTransformInterface]
    scaler: GradScaler | None = None  # For AMP
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16
    step: int = 0

    def to(self, device: torch.device):
        """Move state to device."""
        self.model = self.model.to(device)
        return self


###


CRITERION_COLLECTION = {
    ### changed for torch
    "ce": F.cross_entropy,
    "bce": lambda x, y: F.binary_cross_entropy_with_logits(x, (y > 0).float()).mean(),
    ###
}


class TorchTrainModule(nn.Module):
    """PyTorch training module that wraps model with loss computation."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable = CRITERION_COLLECTION["ce"],
        dtype: str = "float32",
        augmentation: bool = True,
        normalization_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
        normalization_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
        pre_normalized: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.dtype = dtype
        self.augmentation = augmentation
        self.normalization_mean = torch.tensor(normalization_mean).view(1, 1, 1, 3)
        self.normalization_std = torch.tensor(normalization_std).view(1, 1, 1, 3)
        self.pre_normalized = pre_normalized
        self.label_smoothing = label_smoothing

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor, deterministic: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with loss computation."""
        ### changed for torch
        # Normalize the pixel values
        if not self.pre_normalized:
            images = images.float() / 255.0
        images = images.permute(0, 2, 3, 1)

        # Move normalization tensors to same device as images
        if self.normalization_mean.device != images.device:
            self.normalization_mean = self.normalization_mean.to(images.device)
            self.normalization_std = self.normalization_std.to(images.device)

        images = (images - self.normalization_mean) / self.normalization_std

        # Convert to appropriate precision
        if self.dtype == "float16":
            images = images.half()
        elif self.dtype == "bfloat16":
            images = images.bfloat16()
        else:
            images = images.float()

        # Get number of classes from model output
        logits = self.model(images)
        num_classes = logits.shape[-1]

        # Handle labels
        labels = labels.to(dtype=torch.long)
        if labels.dim() == 1:
            # Convert to one-hot if needed
            labels_one_hot = F.one_hot(labels, num_classes).float()
        else:
            labels_one_hot = labels.float()

        # Apply label smoothing if training
        if not deterministic and self.label_smoothing > 0:
            labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        # Compute loss
        if self.criterion == F.cross_entropy:
            # Use original labels for cross entropy
            loss = self.criterion(logits, labels if labels.dim() == 1 else labels.argmax(dim=1))
        else:
            loss = self.criterion(logits, labels_one_hot)

        # Compute accuracies
        with torch.no_grad():
            # Get top predictions
            _, preds = torch.topk(logits, k=min(5, num_classes), dim=-1)

            # Convert labels back to class indices for accuracy computation
            if labels.dim() > 1:
                target_labels = labels.argmax(dim=1)
            else:
                target_labels = labels

            # Compute top-1 accuracy
            acc1 = (preds[:, 0] == target_labels).float()

            metrics = {"loss": loss, "acc1": acc1}

            # Compute top-5 accuracy if we have enough classes
            if num_classes > 5:
                acc5 = (preds == target_labels.unsqueeze(1)).any(dim=1).float()
                metrics["acc5"] = acc5

        return metrics
        ###


def torch_training_step(
    state: TorchTrainState, batch: tuple[torch.Tensor, torch.Tensor], accum_step: int = 0, accum_steps_total: int = 1
) -> Dict[str, torch.Tensor]:
    """Training step function for PyTorch."""
    ### changed for torch
    state.model.train()

    images, labels = batch

    # Zero gradients
    if accum_step == 0:
        state.optimizer.zero_grad()

    # Forward pass with AMP if enabled
    if state.use_amp and state.scaler is not None:
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=state.amp_dtype):
            metrics = state.model(images, labels, deterministic=False)
            loss = metrics["loss"]

        # Backward pass with gradient scaling
        state.scaler.scale(loss).backward()

        # Apply gradient transforms (clipping, etc.) - need to unscale first for clipping
        if accum_step == accum_steps_total - 1:
            state.scaler.unscale_(state.optimizer)
            for transform in state.transforms:
                transform.apply_transform(state.model, state.optimizer)

            # Optimizer step with scaler
            state.scaler.step(state.optimizer)
            state.scaler.update()
    else:
        # Standard forward pass
        metrics = state.model(images, labels, deterministic=False)
        loss = metrics["loss"]

        # Backward pass
        loss.backward()

        # Apply gradient transforms (clipping, etc.)
        if accum_step == accum_steps_total - 1:
            for transform in state.transforms:
                transform.apply_transform(state.model, state.optimizer)

            # Optimizer step
            state.optimizer.step()

    # Scheduler step if available
    if accum_step == accum_steps_total - 1:
        if state.scheduler is not None:
            state.scheduler.step()

        # Update step counter
        state.step += 1

        # Add learning rate to metrics
        current_lr = state.optimizer.param_groups[0]["lr"]
        metrics["learning_rate"] = torch.tensor(current_lr)

    return metrics
    ###


def torch_validation_step(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Validation step function for PyTorch."""
    ### changed for torch
    model.eval()

    with torch.no_grad():
        images, labels = batch

        # Handle invalid labels (set to 0 for computation, will be masked out)
        valid_mask = labels != -1
        labels_clean = torch.where(valid_mask, labels, 0)

        # Forward pass
        with autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=str_dtype_to_torch(model.model.config.dtype),
        ):
            metrics = model(images, labels_clean, deterministic=True)

        # Mask out invalid samples
        for key, value in metrics.items():
            if key != "loss":  # Don't mask loss as it's already computed correctly
                metrics[key] = value * valid_mask.float()

        # Add number of valid samples
        metrics["num_samples"] = valid_mask.float()

    return metrics
    ###


def create_torch_train_state(
    trainer_cfg: Any,
    model_cfg: BaseModelTorch.cfgtype,
    optimizer_cfg: OptimizerInterface.cfgtype,
    device: torch.device,
    use_torch_compile: bool = False,
) -> TorchTrainState:
    """Create PyTorch training state."""
    ### changed for torch
    # Create model
    model = model_cfg.instantiate(BaseModelTorch)

    if use_torch_compile:
        model = torch.compile(model)

    # Wrap model with training module
    train_module = TorchTrainModule(
        model=model,
        criterion=CRITERION_COLLECTION[trainer_cfg.criterion],
        dtype=model_cfg.dtype,
        augmentation=trainer_cfg.augmentation,
        pre_normalized=trainer_cfg.pre_normalized,
        label_smoothing=trainer_cfg.label_smoothing if trainer_cfg.criterion == "ce" else 0,
    )

    # Move to device
    train_module = train_module.to(device)

    # Create optimizer and scheduler
    optimizer_interface = optimizer_cfg.instantiate(OptimizerInterface)
    optimizer, scheduler, transforms = optimizer_interface.create_optimizer_and_scheduler(
        train_module, trainer_cfg.train_steps
    )

    # Set up AMP if dtype is not float32
    use_amp = model_cfg.dtype != "float32"
    scaler = None
    amp_dtype = torch.float16  # default

    if use_amp:
        scaler = GradScaler()
        if model_cfg.dtype == "float16":
            amp_dtype = torch.float16
        elif model_cfg.dtype == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            # For other dtypes, default to float16
            amp_dtype = torch.float16

    # Create training state
    state = TorchTrainState(
        model=train_module,
        optimizer=optimizer,
        scheduler=scheduler,
        transforms=transforms,
        scaler=scaler,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        step=0,
    )

    return state
    ###
