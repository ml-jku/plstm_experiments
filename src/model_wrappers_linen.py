from compoconf import register, ConfigInterface
from dataclasses import dataclass, asdict, field
from plstm.linen.lm_model import pLSTMLMModel as pLSTMLMModel_linen
from plstm.linen.vision_model import pLSTMVisionModel as pLSTMVisionModel_linen
from plstm.config.lm_model import pLSTMLMModelConfig
from plstm.config.vision_model import pLSTMVisionModelConfig
from jax_trainer.interfaces import BaseModelLinen
from flax import nnx
from flax import linen as nn
import jax.numpy as jnp
from typing import Any, Literal

from .vit_external import ViT as ViT_ext
from .vit_external import ViTBase as ViTBase


@register
class pLSTMLMModel(BaseModelLinen, pLSTMLMModel_linen):
    config: pLSTMLMModelConfig

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs = {key: val for key, val in kwargs.items() if key in ("deterministic")}

        return pLSTMLMModel_linen.__call__(self, x, **kwargs)


@register
class pLSTMVisionModel(BaseModelLinen, pLSTMVisionModel_linen):
    config: pLSTMVisionModelConfig

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs = {key: val for key, val in kwargs.items() if key in ("deterministic")}

        return pLSTMVisionModel_linen.__call__(self, x, **kwargs)


@dataclass
class ViTConfig(ViTBase, ConfigInterface):
    _short_name: str = "ViT"
    num_classes: int = 1
    resolution: tuple[int, int] = (224, 224)
    aux: dict[str, Any] = field(default_factory=dict)
    input_shape: tuple[int, int, int] = (224, 224, 3)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    output_shape: tuple[int] = (10,)


@register
class ViTExternal(BaseModelLinen, nn.Module):
    config: ViTConfig

    def setup(self):
        cfg = asdict(self.config)
        del cfg["class_name"]
        del cfg["_short_name"]
        del cfg["num_classes"]
        del cfg["aux"]
        del cfg["input_shape"]
        del cfg["resolution"]
        del cfg["dtype"]
        del cfg["output_shape"]

        self.model = ViT_ext(**cfg)

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        return self.model(x, deterministic=deterministic)
