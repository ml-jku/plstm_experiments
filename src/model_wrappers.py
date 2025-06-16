from compoconf import register, ConfigInterface, parse_config
from dataclasses import dataclass, asdict, field
from plstm.nnx.lm_model import pLSTMLMModel as pLSTMLMModel_nnx
from plstm.nnx.vision_model import pLSTMVisionModel as pLSTMVisionModel_nnx
from plstm.config.lm_model import pLSTMLMModelConfig
from plstm.config.vision_model import pLSTMVisionModelConfig
from jax_trainer.interfaces import BaseModelNNX
from flax import nnx
import jax.numpy as jnp
from typing import Any, Literal

from .vit_external import ViT as ViT_ext
from .vit_external import ViTBase as ViTBase
from .model_wrappers_linen import ViTConfig


@register
class pLSTMLMModel(BaseModelNNX, pLSTMLMModel_nnx):
    config_class = pLSTMLMModelConfig

    def __init__(self, config: pLSTMLMModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        pLSTMLMModel_nnx.__init__(self, config, rngs=rngs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs = {key: val for key, val in kwargs.items() if key in ("deterministic")}
        return pLSTMLMModel_nnx.__call__(self, x, **kwargs)


@register
class pLSTMVisionModel(BaseModelNNX, pLSTMVisionModel_nnx):
    config_class = pLSTMVisionModelConfig

    def __init__(self, config: pLSTMVisionModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        pLSTMVisionModel_nnx.__init__(self, config, rngs=rngs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs = {key: val for key, val in kwargs.items() if key in ("deterministic")}
        return pLSTMVisionModel_nnx.__call__(self, x, **kwargs)


@register
class ViTExternal(BaseModelNNX, nnx.Module):
    config_class = ViTConfig

    def __init__(self, config: ViTConfig, rngs: nnx.Rngs):
        self.config = config
        cfg = asdict(config)
        del cfg["class_name"]
        del cfg["_short_name"]
        del cfg["num_classes"]
        del cfg["aux"]
        del cfg["input_shape"]
        model = ViT_ext(**cfg)
        self.model = nnx.bridge.ToNNX(model, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.model(x)
