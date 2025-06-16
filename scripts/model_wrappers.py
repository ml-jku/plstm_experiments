from compoconf import register
from plstm.nnx.lm_model import pLSTMLMModel as pLSTMLMModel_jax
from plstm.nnx.vision_model import pLSTMVisionModel as pLSTMVisionModel_jax
from plstm.config.lm_model import pLSTMLMModelConfig
from plstm.config.vision_model import pLSTMVisionModelConfig
from jax_trainer.interfaces import BaseModelNNX
from flax import nnx
import jax.numpy as jnp


@register
class pLSTMLMModel(BaseModelNNX, pLSTMLMModel_jax):
    config_class = pLSTMLMModelConfig

    def __init__(self, config: pLSTMLMModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        pLSTMLMModel_jax.__init__(self, config, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return pLSTMLMModel_jax.__call__(self, x)


@register
class pLSTMVisionModel(BaseModelNNX, pLSTMVisionModel_jax):
    config_class = pLSTMVisionModelConfig

    def __init__(self, config: pLSTMVisionModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        pLSTMVisionModel_jax.__init__(self, config, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return pLSTMVisionModel_jax.__call__(self, x)
