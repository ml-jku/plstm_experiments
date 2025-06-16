# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations


from typing import Any, Dict, TypeVar, Union, List, Tuple, Set, Iterable
import copy
import dataclasses


import argparse
import json
import os
import re
import threading
from collections import defaultdict

import flax
from jax.sharding import Mesh
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import webdataset as wds
from chex import Array, ArrayTree
from jax.tree_util import DictKey


class AverageMeter:
    def __init__(self, use_latest: list[str] = []):
        self.buffer = defaultdict(list)
        self.use_latest = use_latest

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self.buffer[k].append(v)

    def summary(self, prefix: str = "") -> dict[str, float]:
        buffer = {k: np.array(v) for k, v in self.buffer.items()}
        self.buffer.clear()

        return {f"{prefix}{k}": v[-1] if k in self.use_latest else np.mean(v) for k, v in buffer.items()}


def save_checkpoint_in_background(config: Any, params_bytes: bytes, postfix: str = "last"):
    def thread_fn():
        filename = os.path.join(config.trainer.log_dir, f"{config.trainer.name}-{postfix}.msgpack")
        with wds.gopen(filename, "wb") as fp:
            fp.write(params_bytes)

    threading.Thread(target=thread_fn).start()


class Mixup(nn.Module):
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0

    def apply_mixup(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.mixup_alpha,) * 2)
        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = ratio * images + (1 - ratio) * images[randperm]
        labels = ratio * labels + (1 - ratio) * labels[randperm]
        return images, labels

    def apply_cutmix(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.cutmix_alpha,) * 2)
        image_mask = self.random_bounding_box(ratio, images.shape[2], images.shape[1])
        label_mask = image_mask.mean((1, 2))

        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = image_mask * images + (1 - image_mask) * images[randperm]
        labels = label_mask * labels + (1 - label_mask) * labels[randperm]
        return images, labels

    def random_bounding_box(self, ratio: Array, width: int, height: int) -> Array:
        size = (1 - ratio) ** 0.5
        xstart, ystart = jax.random.uniform(self.make_rng("mixup"), (2,))
        xrange, yrange = jnp.linspace(0, 1, width), jnp.linspace(0, 1, height)

        xmask = (xstart - 0.5 * size <= xrange) & (xrange < xstart + 0.5 * size)
        ymask = (ystart - 0.5 * size <= yrange) & (yrange < ystart + 0.5 * size)
        return ~(xmask[None, None, :, None] & ymask[None, :, None, None])

    def __call__(self, images: Array, labels: Array) -> tuple[Array, Array]:
        if self.mixup_alpha == 0 and self.cutmix_alpha == 0:
            return images, labels
        if self.mixup_alpha > 0 and self.cutmix_alpha == 0:
            return self.apply_mixup(images, labels)
        if self.mixup_alpha == 0 and self.cutmix_alpha > 0:
            return self.apply_cutmix(images, labels)

        # If both mixup and cutmix are enabled, only one operation will be selected and
        # applied. Since jax does not support conditional branching on JIT, mixup and
        # cutmix are performed first and only one output will be selected.
        images1, labels1 = self.apply_mixup(images, labels)
        images2, labels2 = self.apply_cutmix(images, labels)

        cond = jax.random.uniform(self.make_rng("mixup")) > 0.5
        return jnp.where(cond, images1, images2), jnp.where(cond, labels1, labels2)


class MixupNNX(nnx.Module):
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        rng_collection: str = "mixup",
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.rng_collection = rng_collection
        self.rngs = rngs
        self.mesh = mesh

    def apply_mixup(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.mixup_alpha,) * 2)
        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = ratio * images + (1 - ratio) * images[randperm]
        labels = ratio * labels + (1 - ratio) * labels[randperm]
        return images, labels

    def apply_cutmix(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.cutmix_alpha,) * 2)
        image_mask = self.random_bounding_box(ratio, images.shape[2], images.shape[1])
        label_mask = image_mask.mean((1, 2))

        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = image_mask * images + (1 - image_mask) * images[randperm]
        labels = label_mask * labels + (1 - label_mask) * labels[randperm]
        return images, labels

    def random_bounding_box(self, ratio: Array, width: int, height: int) -> Array:
        size = (1 - ratio) ** 0.5
        xstart, ystart = jax.random.uniform(self.make_rng("mixup"), (2,))
        xrange, yrange = jnp.linspace(0, 1, width), jnp.linspace(0, 1, height)

        xmask = (xstart - 0.5 * size <= xrange) & (xrange < xstart + 0.5 * size)
        ymask = (ystart - 0.5 * size <= yrange) & (yrange < ystart + 0.5 * size)
        return ~(xmask[None, None, :, None] & ymask[None, :, None, None])

    def __call__(self, images: Array, labels: Array) -> tuple[Array, Array]:
        if self.mixup_alpha == 0 and self.cutmix_alpha == 0:
            return images, labels
        if self.mixup_alpha > 0 and self.cutmix_alpha == 0:
            return self.apply_mixup(images, labels)
        if self.mixup_alpha == 0 and self.cutmix_alpha > 0:
            return self.apply_cutmix(images, labels)

        # If both mixup and cutmix are enabled, only one operation will be selected and
        # applied. Since jax does not support conditional branching on JIT, mixup and
        # cutmix are performed first and only one output will be selected.
        images1, labels1 = self.apply_mixup(images, labels)
        images2, labels2 = self.apply_cutmix(images, labels)

        cond = jax.random.uniform(self.make_rng("mixup")) > 0.5
        return jnp.where(cond, images1, images2), jnp.where(cond, labels1, labels2)


def fixed_sincos2d_embeddings(ncols: int, nrows: int, dim: int) -> Array:
    freqs = 1 / (10000 ** jnp.linspace(0, 1, dim // 4))
    x = jnp.outer(jnp.arange(0, nrows, dtype=jnp.float32), freqs)
    y = jnp.outer(jnp.arange(0, ncols, dtype=jnp.float32), freqs)

    x = jnp.broadcast_to(x[None, :, :], (ncols, nrows, dim // 4))
    y = jnp.broadcast_to(y[:, None, :], (ncols, nrows, dim // 4))
    return jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=2)


def modified_lamb(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mask: optax.MaskOrFn = None,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay=weight_decay, mask=mask),
        # Change to use trust ratio on weight decay parameters only.
        optax.masked(optax.scale_by_trust_ratio(), mask=mask),
        optax.scale_by_learning_rate(learning_rate),
    )


def get_layer_index_fn(path: tuple[DictKey, ...], _: Any, num_layers: int = 12) -> int:
    if path[0].key == "model" and path[1].key.startswith("layer_"):
        return int(re.match(r"layer_(\d+)", path[1].key).group(1)) + 1
    if path[0].key == "model" and path[1].key == "embed":
        return 0
    return num_layers


def load_pretrained_params(trainer_cfg: Any, model_cfg: Any, params: ArrayTree) -> ArrayTree:
    with wds.gopen(trainer_cfg.pretrained_ckpt) as fp:
        new_params = flax.serialization.msgpack_restore(fp.read())

    # The positional embeddings will be resized when there is a difference in image
    # resolutions between pretraining and finetuning stage.
    if (
        model_cfg.use_pos_embed
        and new_params["model"]["pos_embed"]["embed"].shape != params["model"]["pos_embed"]["embed"].shape
    ):
        new_params["model"]["pos_embed"]["embed"] = jax.image.resize(
            new_params["model"]["pos_embed"]["embed"],
            params["model"]["pos_embed"]["embed"].shape,
            method="bicubic",
        )

    return new_params


class LimitIterable:
    def __init__(self, it: Iterable, limit: int = None):
        self.it = it
        self._it = None
        self.limit = limit
        self.count = 0

    def __len__(self):
        real_len = len(self.it)
        return min(real_len, self.limit) if self.limit is not None else real_len

    def __iter__(self):
        self._it = iter(self.it)
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        if self.limit is not None and self.count == self.limit:
            raise StopIteration
        return next(self._it)


T = TypeVar("T")


def apply_override(obj: T, override: Dict[str, Any], enforce: bool = False, exact_match: bool = True) -> T:
    """
    Recursively applies values from an override dictionary to an object or dictionary.

    This function can modify attributes of an object or keys in a dictionary based on the
    provided override dictionary. It handles nested structures by recursively applying
    overrides to nested objects or dictionaries.

    Parameters
    ----------
    obj : Any
        The target object or dictionary to be modified.
    override : Dict[str, Any]
        Dictionary containing the values to override in the target object.
    enforce : bool, default=False
        If True, will force setting attributes/keys even if they don't exist in the original object.
        If False, will only override existing attributes/keys.
    exact_match : bool, default=True
        If True and enforce is False, will raise ValueError when a key doesn't exist.
        If False, will silently ignore keys that don't exist.

    Returns
    -------
    T
        The modified object with overrides applied.

    Examples
    --------
    >>> class Config:
    ...     def __init__(self):
    ...         self.name = "default"
    ...         self.params = {"learning_rate": 0.01, "batch_size": 32}
    ...     def __repr__(self):
    ...         return f"Config(name='{self.name}', params={self.params})"
    >>> config = Config()
    >>> override = {"name": "custom", "params": {"learning_rate": 0.001}}
    >>> apply_override(config, override)
    Config(name='custom', params={'learning_rate': 0.001, 'batch_size': 32})

    >>> # Dictionary example
    >>> base_dict = {"model": {"type": "cnn", "layers": 3}, "training": {"epochs": 10}}
    >>> override_dict = {"model": {"layers": 5}, "training": {"batch_size": 64}}
    >>> apply_override(base_dict, override_dict, enforce=True)
    {'model': {'type': 'cnn', 'layers': 5}, 'training': {'epochs': 10, 'batch_size': 64}}

    >>> # Using enforce=True to add new attributes
    >>> config = Config()
    >>> override = {"new_param": "value", "params": {"optimizer": "adam"}}
    >>> apply_override(config, override, enforce=True)
    Config(name='default', params={'learning_rate': 0.01, 'batch_size': 32, 'optimizer': 'adam'})
    >>> config.new_param
    'value'

    >>> # Using exact_match=False to ignore non-existent keys
    >>> config = Config()
    >>> override = {"unknown": "value"}
    >>> apply_override(config, override, exact_match=False)
    Config(name='default', params={'learning_rate': 0.01, 'batch_size': 32})

    >>> # Using exact_match=True (default) raises ValueError for non-existent keys
    >>> config = Config()
    >>> override = {"unknown": "value"}
    >>> try:
    ...     apply_override(config, override)
    ... except ValueError:
    ...     print("ValueError raised as expected")
    ValueError raised as expected
    """
    if isinstance(override, dict):
        for sub, val in override.items():
            if hasattr(obj, sub):
                setattr(obj, sub, apply_override(getattr(obj, sub), val, enforce=enforce, exact_match=exact_match))
            else:
                try:
                    if sub in obj or enforce:
                        if sub not in obj:
                            obj[sub] = val
                        else:
                            obj[sub] = apply_override(obj[sub], val, enforce=enforce, exact_match=exact_match)
                except TypeError:
                    if enforce:
                        setattr(obj, sub, val)
                    elif exact_match:
                        raise ValueError
        return obj
    else:
        return override


def deepcopy_with_dataclasses(obj: Any) -> Any:
    """
    Performs a deep copy of an object while preserving dataclass types.

    Standard copy.deepcopy() can convert nested dataclasses to dictionaries.
    This function ensures that dataclass instances remain as dataclasses
    throughout the copying process.

    Parameters
    ----------
    obj : Any
        The object to be deep copied.

    Returns
    -------
    Any
        A deep copy of the input object with preserved dataclass types.

    Examples
    --------
    >>> from dataclasses import dataclass, field
    >>> @dataclass
    ... class NestedConfig:
    ...     value: int = 42
    ...
    >>> @dataclass
    ... class Config:
    ...     name: str = "default"
    ...     nested: NestedConfig = field(default_factory=NestedConfig)
    ...
    >>> original = Config()
    >>> copied = deepcopy_with_dataclasses(original)
    >>> copied.name = "modified"
    >>> copied.nested.value = 100
    >>> original.name, original.nested.value
    ('default', 42)
    >>> copied.name, copied.nested.value
    ('modified', 100)
    >>> isinstance(copied.nested, NestedConfig)
    True
    """
    # Handle None
    if obj is None:
        return None

    # Handle dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Create a new instance of the same dataclass type
        result = copy.copy(obj)
        # Deep copy all fields
        for field in dataclasses.fields(obj):
            field_name = field.name
            field_value = getattr(obj, field_name)
            setattr(result, field_name, deepcopy_with_dataclasses(field_value))
        return result

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: deepcopy_with_dataclasses(v) for k, v in obj.items()}

    # Handle lists
    elif isinstance(obj, list):
        return [deepcopy_with_dataclasses(item) for item in obj]

    # Handle tuples
    elif isinstance(obj, tuple):
        return tuple(deepcopy_with_dataclasses(item) for item in obj)

    # Handle sets
    elif isinstance(obj, set):
        return {deepcopy_with_dataclasses(item) for item in obj}

    # For other types, use standard deepcopy
    else:
        return copy.deepcopy(obj)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
