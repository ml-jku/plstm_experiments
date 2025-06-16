from __future__ import annotations

import argparse
from functools import partial
from typing import Callable, Dict, Any, Tuple

import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import Array, ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from jax.tree_util import tree_map_with_path

from .simple_dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .utils import Mixup, get_layer_index_fn, load_pretrained_params, modified_lamb
from jax_trainer.interfaces import BaseModelLinen, BaseModelNNX
from jax_trainer.optimizer import OptimizerInterface
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from dataclasses import dataclass, field
from compoconf import ConfigInterface, register_interface, register, RegistrableConfigInterface
from .utils import AverageMeter


# num_devices = jax.device_count()
# state_axes = nnx.StateAxes({nnx.Param: None, nnx.RngState: 0})


@dataclass
class VisionTrainerConfig(ConfigInterface):
    init_seed: int = 42
    log_dir: str = "./outputs"
    name: str = ""
    project: str = ""
    logging: bool = True
    log_interval: int = 100
    eval_interval: int = 100
    train_batch_size: int = 128
    train_steps: int = 1000
    train_epochs: int = -1  # may be used to calc train_steps
    grad_accum: int = 1
    criterion: str = "ce"
    model_seed: int = 43
    pretrained_ckpt: str | None = None

    mixup: float = 0.8
    mixup_seed: int = 44
    dropout_seed: int = 45
    cutmix: float = 1.0
    label_smoothing: float = 0.0
    augmentation: bool = True
    pre_normalized: bool = False


@register
@register_interface
class VisionTrainer(RegistrableConfigInterface):
    config: VisionTrainerConfig

    def __init__(self, config: VisionTrainerConfig):
        self.config = config


CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
}


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    def split_rngs(self) -> tuple[ArrayTree, ArrayTree]:
        mixup_rng, new_mixup_rng = jax.random.split(self.mixup_rng)
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)

        rngs = {"mixup": mixup_rng, "dropout": dropout_rng}
        updates = {"mixup_rng": new_mixup_rng, "dropout_rng": new_dropout_rng}
        return rngs, updates

    def replicate(self) -> TrainState:
        """Replicate the state across all devices."""
        return flax.jax_utils.replicate(self).replace(
            mixup_rng=shard_prng_key(self.mixup_rng),
            dropout_rng=shard_prng_key(self.dropout_rng),
        )


class TrainModule(nn.Module):
    model: Any
    mixup: Mixup
    label_smoothing: float = 0.0
    criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"]
    dtype: str = "float32"
    augmentation: bool = True
    normalization_mean: tuple[float, float, float] = field(default_factory=lambda: IMAGENET_DEFAULT_MEAN)
    normalization_std: tuple[float, float, float] = field(default_factory=lambda: IMAGENET_DEFAULT_STD)
    pre_normalized: bool = False

    def __call__(self, images: Array, labels: Array, deterministic: bool = True) -> ArrayTree:
        # Normalize the pixel values
        if not self.pre_normalized:
            images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
        else:
            images = jnp.moveaxis(images, 1, 3)

        images = (images - np.asarray(self.normalization_mean)) / np.asarray(self.normalization_std)

        # Convert to appropriate precision
        dtype = getattr(jnp, self.dtype)
        images = images.astype(dtype)
        num_classes = self.model.config.output_shape[0]
        labels = nn.one_hot(labels, num_classes) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)

        if not deterministic and self.augmentation:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            images, labels = self.mixup(images, labels)

        loss = self.criterion((logits := self.model(images, deterministic=deterministic)), labels)
        labels = labels == labels.max(-1, keepdims=True)

        # Instead of directly comparing the maximum classes of predicted logits with the
        # given one-hot labels, we will check if the predicted classes are within the
        # label set. This approach is equivalent to traditional methods in single-label
        # classification and also supports multi-label tasks.
        preds = jax.lax.top_k(logits, k=min(5, num_classes))[1]
        accs = jnp.take_along_axis(labels, preds, axis=-1)
        metrics = {"loss": loss, "acc1": accs[:, 0]}
        if num_classes > 5:
            metrics["acc5"] = accs.any(-1)
        return metrics


class TrainModuleNNX(nnx.Module):
    model: Any
    mixup: Mixup
    label_smoothing: float = 0.0

    def __init__(
        self,
        model: nnx.Module,
        mixup: Mixup,
        label_smoothing: float = 0.0,
        criterion: Callable[[Array, Array], Array] = CRITERION_COLLECTION["ce"],
        dtype: str = "float32",
        augmentation: bool = True,
        normalization_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
        normalization_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
        pre_normalized: bool = False,
        rngs: nnx.Rngs | None = None,
    ):
        self.model = model
        self.mixup = nnx.bridge.ToNNX(mixup, rngs=rngs)
        self.label_smoothing = label_smoothing
        self.criterion = criterion
        self.dtype = dtype
        self.augmentation = augmentation
        self.normalization_mean = normalization_mean
        self.normalization_std = (normalization_std,)
        self.pre_normalized = pre_normalized

    def __call__(self, images: Array, labels: Array, deterministic: bool = True) -> ArrayTree:
        # Normalize the pixel values
        if not self.pre_normalized:
            images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
        else:
            images = jnp.moveaxis(images, 1, 3)
        images = (images - np.asarray(IMAGENET_DEFAULT_MEAN)) / np.asarray(IMAGENET_DEFAULT_STD)

        # Convert to appropriate precision
        dtype = getattr(jnp, self.dtype)
        images = images.astype(dtype)
        num_classes = self.model.config.output_shape[0]
        labels = nn.one_hot(labels, self.model.config.output_shape[0]) if labels.ndim == 1 else labels
        labels = labels.astype(jnp.float32)

        if not deterministic and self.augmentation:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            images, labels = self.mixup(images, labels)

        loss = self.criterion((logits := self.model(images, deterministic=deterministic)), labels)
        labels = labels == labels.max(-1, keepdims=True)

        # Instead of directly comparing the maximum classes of predicted logits with the
        # given one-hot labels, we will check if the predicted classes are within the
        # label set. This approach is equivalent to traditional methods in single-label
        # classification and also supports multi-label tasks.
        preds = jax.lax.top_k(logits, k=min(5, num_classes))[1]
        accs = jnp.take_along_axis(labels, preds, axis=-1)
        metrics = {"loss": loss, "acc1": accs[:, 0]}
        if num_classes > 5:
            metrics["acc5"]: accs.any(-1)
        return metrics


@partial(jax.pmap, axis_name="data", donate_argnums=0)
def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    """Training step function with data parallelism across multiple devices.

    This version uses pmap to distribute computation across all available devices.
    Each device processes a shard of the batch, and gradients are synchronized
    across devices using pmean.
    """

    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics = state.apply_fn({"params": params}, *batch, deterministic=False, rngs=rngs)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics

    def update_fn(state: TrainState) -> tuple[TrainState, ArrayTree]:
        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_util.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        grad_norm = optax.global_norm(grads)
        return state.apply_gradients(
            grads=jax.lax.pmean(grads, axis_name="data"),
            grad_accum=jax.tree_util.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        ), {"grad_norm": grad_norm}

    rngs, updates = state.split_rngs()
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics = jax.lax.pmean(metrics, axis_name="data")

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        grads = jax.lax.pmean(grads, axis_name="data")
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        grad_metrics = {"grad_norm": grad_norm}
    else:
        state = state.replace(
            grad_accum=jax.tree_util.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state, grad_metrics = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: (x, {"grad_norm": jnp.zeros([1])}), state
        )
    # needs to be adapted for optimizer structure possibly
    hyperparams = {}
    for inner_state in state.opt_state.inner_state:
        if hasattr(inner_state, "hyperparams"):
            hyperparams.update(**inner_state.hyperparams)
    return state.replace(**updates), metrics | state.opt_state.hyperparams | hyperparams | grad_metrics


@partial(jax.pmap, axis_name="data")
def validation_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
    """Validation step function with data parallelism across multiple devices.

    This version uses pmap to distribute computation across all available devices.
    Each device processes a shard of the batch, and metrics are aggregated
    across devices using psum.
    """
    metrics = state.apply_fn(
        {"params": state.params},
        images=batch[0],
        labels=jnp.where(batch[1] != -1, batch[1], 0),
        deterministic=True,
    )
    metrics["num_samples"] = batch[1] != -1
    metrics = jax.tree_util.tree_map(lambda x: (x * (batch[1] != -1)).sum(), metrics)
    return jax.lax.psum(metrics, axis_name="data")


def create_train_state(
    trainer_cfg: Any, model_cfg: BaseModelLinen.cfgtype, optimizer_cfg: OptimizerInterface.cfgtype
) -> TrainState:
    """Create training state optimized for GPU with data parallelism."""
    # Set precision based on args
    dtype = model_cfg.dtype

    model = model_cfg.instantiate(BaseModelLinen)
    module = TrainModule(
        model=model,
        mixup=Mixup(trainer_cfg.mixup, trainer_cfg.cutmix),
        label_smoothing=trainer_cfg.label_smoothing if trainer_cfg.criterion == "ce" else 0,
        criterion=CRITERION_COLLECTION[trainer_cfg.criterion],
        dtype=dtype,
        augmentation=trainer_cfg.augmentation,
        pre_normalized=trainer_cfg.pre_normalized,
    )

    # Initialize the model weights with dummy inputs
    example_inputs = {
        "images": jnp.zeros((1, 3, *model_cfg.resolution), dtype=jnp.uint8),
        "labels": jnp.zeros((1,), dtype=jnp.int32),
    }
    init_rngs = {"params": jax.random.PRNGKey(trainer_cfg.model_seed)}
    print(module.tabulate(init_rngs, **example_inputs))

    params = module.init(init_rngs, **example_inputs)["params"]
    if trainer_cfg.pretrained_ckpt is not None:
        params = load_pretrained_params(trainer_cfg, model_cfg, params)
    if trainer_cfg.grad_accum > 1:
        grad_accum = jax.tree_util.tree_map(jnp.zeros_like, params)

    # Create learning rate scheduler and optimizer with gradient clipping
    tx = optimizer_cfg.instantiate(OptimizerInterface)

    # Create the training state
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        mixup_rng=jax.random.PRNGKey(trainer_cfg.mixup_seed + jax.process_index()),
        dropout_rng=jax.random.PRNGKey(trainer_cfg.dropout_seed + jax.process_index()),
        micro_step=0,
        micro_in_mini=trainer_cfg.grad_accum,
        grad_accum=grad_accum if trainer_cfg.grad_accum > 1 else None,
    )


class GradAccumOptimizer(nnx.Optimizer):
    def __init__(self, model: nnx.Module, tx: optax.GradientTransformation, accumulation_steps: int, wrt=nnx.Param):
        """
        model: your NNX Module
        tx:    an Optax transform (e.g. optax.adam(...))
        accumulation_steps: # of micro‑batches to accumulate
        wrt:   which Variable types to optimize (see nnx.Optimizer)
        """
        super().__init__(model, tx, wrt=wrt)  # :contentReference[oaicite:0]{index=0}
        self.accum_steps = accumulation_steps
        self._grad_buffer = None
        self._counter = 0

    def update(self, *, grads, **kwargs):
        # lazy‑init buffer to zeros with the same structure as grads
        if self._grad_buffer is None:
            self._grad_buffer = jax.tree_map(lambda g: jnp.zeros_like(g), grads)

        # accumulate
        self._grad_buffer = jax.tree_map(lambda buf, g: buf + g, self._grad_buffer, grads)
        self._counter += 1

        # if we haven’t hit the threshold, just return self (no step)
        if self._counter < self.accum_steps:
            return self

        # compute average grads
        avg_grads = jax.tree_map(lambda buf: buf / self.accum_steps, self._grad_buffer)

        # perform the actual update
        new_state = super().update(grads=avg_grads, **kwargs)

        # reset buffer and counter
        self._grad_buffer = jax.tree_map(lambda g: jnp.zeros_like(g), avg_grads)
        self._counter = 0

        return new_state

    def zero_grad(self):
        self._grad_buffer = None


# @nnx.split_rngs(splits=num_devices, only=("dropout", "mixup"))
# @nnx.pmap(in_axes=(state_axes, {"images": 0, "labels": 0}), axis_name="data")
@nnx.jit
def training_nnx_step(state: nnx.Module, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    """Training step function with data parallelism across multiple devices.

    This version uses pmap to distribute computation across all available devices.
    Each device processes a shard of the batch, and gradients are synchronized
    across devices using pmean.
    """

    def loss_fn(m) -> ArrayTree:
        metrics = m(*batch)
        metrics = jax.tree_util(jnp.mean, metrics)
        return metrics["loss"], metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state.model)

    return metrics | state.step.hyperparams


# @partial(jax.pmap, axis_name="data")
@nnx.jit
def validation_nnx_step(state: nnx.Module, batch: ArrayTree) -> ArrayTree:
    """Validation step function with data parallelism across multiple devices.

    This version uses pmap to distribute computation across all available devices.
    Each device processes a shard of the batch, and metrics are aggregated
    across devices using psum.
    """
    metrics = state.model(*batch)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)

    return metrics


def create_train_nnx_state(
    trainer_cfg: Any, model_cfg: BaseModelNNX.cfgtype, optimizer_cfg: OptimizerInterface.cfgtype, mesh: Mesh
) -> nnx.Module:
    """Create training state optimized for GPU with data parallelism."""
    # Set precision based on args
    dtype = model_cfg.dtype

    init_rngs = nnx.Rngs(
        trainer_cfg.init_seed,
        params=jax.random.PRNGKey(trainer_cfg.model_seed),
        mixup=jax.random.PRNGKey(trainer_cfg.mixup_seed),
        dropout=jax.random.PRNGKey(trainer_cfg.dropout_seed),
    )
    model = model_cfg.instantiate(BaseModelNNX, rngs=init_rngs)

    module = TrainModuleNNX(
        model=model,
        mixup=Mixup(trainer_cfg.mixup, trainer_cfg.cutmix),
        label_smoothing=trainer_cfg.label_smoothing if trainer_cfg.criterion == "ce" else 0,
        criterion=CRITERION_COLLECTION[trainer_cfg.criterion],
        dtype=dtype,
        augmentation=trainer_cfg.augmentation,
        pre_normalized=trainer_cfg.pre_normalized,
        rngs=init_rngs,
    )

    # Initialize the model weights with dummy inputs
    example_inputs = {
        "images": jnp.zeros(
            (trainer_cfg.train_batch_size // trainer_cfg.grad_accum, 3, *model_cfg.resolution), dtype=jnp.uint8
        ),
        "labels": jnp.zeros((trainer_cfg.train_batch_size // trainer_cfg.grad_accum,), dtype=jnp.int32),
    }
    # not supported yet
    # print(nnx.tabulate(module, **example_inputs))

    nnx.bridge.lazy_init(module, **example_inputs)
    params = nnx.state(module, nnx.Param)
    if trainer_cfg.pretrained_ckpt is not None:
        params = load_pretrained_params(trainer_cfg, model_cfg, params)
        nnx.update(module, params)

    tx = optimizer_cfg.instantiate(OptimizerInterface)

    if trainer_cfg.grad_accum > 1:
        opt = GradAccumOptimizer(module, tx, trainer_cfg.grad_accum)
    else:
        opt = nnx.Optimizer(module, tx)

    # graphdef, param, rng_keys, rng_counts, opt_state = nnx.split(
    #     state, nnx.Param, nnx.RngKey, nnx.RngCount, nnx.optimizer.OptState
    # )
    # replicated param and opt_state for DDP
    # dist_state = {
    #     "param": param,
    #     "rng_keys": jax.tree_util.tree_map(
    #         shard_prng_key, rng_keys, is_leaf=lambda x: isinstance(x, nnx.VariableState)
    #     ),
    #     "rng_counts": rng_counts,
    #     "opt_state": opt_state,
    # }
    # param_spec = jax.tree_util.tree_map(
    #     lambda x: NamedSharding(mesh, P()), param, is_leaf=lambda x: isinstance(x, nnx.VariableState)
    # )
    # opt_state_spec = jax.tree_util.tree_map(
    #     lambda x: NamedSharding(mesh, P()), opt_state, is_leaf=lambda x: isinstance(x, nnx.VariableState)
    # )
    # # replicate rngs for dropout and augmentation
    # rng_keys_spec = jax.tree_util.tree_map(
    #     lambda x: NamedSharding(mesh, P("data")), rng_keys, is_leaf=lambda x: isinstance(x, nnx.VariableState)
    # )
    # rng_counts_spec = jax.tree_util.tree_map(
    #     lambda x: NamedSharding(mesh, P("data")), rng_counts, is_leaf=lambda x: isinstance(x, nnx.VariableState)
    # )

    # sharded_state = jax.lax.with_sharding_constraint(
    #     dist_state, {"param": param_spec, "rngs": rngs_spec, "opt_state": opt_state_spec}
    # )
    # state.update(sharded_state["param"], sharded_state["rngs"], sharded_state["opt_state"])

    state = nnx.state(opt)

    def _annotate(vs: nnx.VariableState):
        if vs.type == nnx.RngKey:
            # shard the key (a length‐2 key‐array) across the 'data' axis:
            idx = jax.lax.axis_index("batch")
            # vs.value = random.fold_in(vs.value, idx)
            vs.sharding = P("batch")
        elif vs.type == nnx.RngCount:
            vs.sharding = None
        else:
            vs.sharding = P()
        # if vs.value is not None and len(vs.value.shape) == 0:
        #     vs.value = vs.value.reshape((1,))
        return vs

    state = jax.tree_map(_annotate, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))
    pspecs = nnx.get_partition_spec(state)
    sharded_state = nnx.with_sharding_constraint(state, pspecs, mesh=mesh)

    nnx.update(opt, sharded_state)

    return opt
