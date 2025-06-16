from dataclasses import dataclass, field, fields
from typing import Any, Dict, Tuple, Literal
import math

### changed for torch
import torch
import torch.nn as nn
import torch.nn.functional as F
###

from compoconf import ConfigInterface, register_interface, register, RegistrableConfigInterface, assert_check_literals

from plstm.torch.vision_model import pLSTMVisionModel as pLSTMVisionModel_torch
from plstm.config.vision_model import pLSTMVisionModelConfig


@dataclass
class BaseModelTorchConfig(ConfigInterface):
    """Base configuration for PyTorch models."""

    dtype: str = "float32"
    resolution: Tuple[int, int] = field(default_factory=lambda: (224, 224))
    output_shape: Tuple[int] = field(default_factory=lambda: (1000,))


@register_interface
class BaseModelTorch(nn.Module, RegistrableConfigInterface):
    """Base interface for PyTorch models."""

    def __init__(self, config: BaseModelTorchConfig):
        self.config = config
        nn.Module.__init__(self)


@register
class pLSTMVisionModel(pLSTMVisionModel_torch, BaseModelTorch):
    config: pLSTMVisionModelConfig


@dataclass
class ViLConfig(ConfigInterface):
    _short_name: str = "ViL"
    dim: int = 192
    depth: int = 12
    patch_size: int = 16
    input_shape: tuple[int, int, int] = (224, 224, 3)
    output_shape: tuple[int] = (2,)
    drop_path_rate: float = 0.0
    dtype: str = "bfloat16"

    aux: dict[str, Any] = field(default_factory=dict)
    resolution: tuple[int, int] = (224, 224)


@register
class ViL(BaseModelTorch):
    config: ViLConfig

    def __init__(self, config: ViLConfig):
        super().__init__(config)
        self.config = config
        self.model = torch.hub.load(
            "nx-ai/vision-lstm",
            "VisionLSTM2",
            dim=config.dim,
            depth=config.depth,
            patch_size=config.patch_size,
            input_shape=(config.input_shape[2], config.input_shape[0], config.input_shape[1]),
            output_shape=config.output_shape,
            drop_path_rate=config.drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.permute(0, 3, 1, 2))


@dataclass
class Mamba2DConfig(ConfigInterface):
    _short_name: str = "Mamba2d"
    input_dim: int = 192
    input_shape: tuple[int, int, int] = (224, 224, 3)
    output_shape: tuple[int] = (2,)
    dtype: str = "bfloat16"
    patch_size: int = 16

    aux: dict[str, Any] = field(default_factory=dict)
    resolution: tuple[int, int] = (224, 224)


@register
class Mamba2D(BaseModelTorch):
    config: Mamba2DConfig

    def __init__(self, config: Mamba2DConfig):
        super().__init__(config)
        self.config = config
        import os
        import sys

        sys.path.append(os.path.split(os.path.abspath(__file__))[0] + "/../../Mamba2D")
        from models.mamba2d import Mamba2DBackbone

        self.model = Mamba2DBackbone(
            in_channels=3,
            channel_last=True,
            embed_dim=[config.input_dim // 2, config.input_dim, config.input_dim, config.input_dim],
            drop_path_rate=0.0,
        )

        self.norm = nn.LayerNorm(normalized_shape=(config.input_dim,))
        self.head = nn.Linear(config.input_dim, config.output_shape[0])

    def forward(self, x: torch.Tensor):
        x = self.model(x.permute(0, 3, 1, 2))
        x = self.norm(x.mean((-2, -1)))
        return self.head(x)


@dataclass
class TwoDMambaConfig(ConfigInterface):
    _short_name: str = "2DMamba"
    input_dim: int = 96
    input_shape: tuple[int, int, int] = (224, 224, 3)
    output_shape: tuple[int] = (2,)
    dtype: str = "float32"
    patch_size: int = 16

    depths: tuple[int, int, int, int] = (2, 2, 5, 2)
    dims: tuple[int, int, int, int] = (96, 96, 96, 96)
    ssm_d_state: int = 1
    ssm_dt_rank: Literal["auto"] | int = "auto"
    ssm_ratio: float = 2.0
    ssm_conv: int = 3
    ssm_conv_bias: bool = False
    ssm_forward_type: Literal["v05_noz"] = "v05_noz"
    mlp_ratio: float = 4.0
    downsample: Literal["v3"] = "v3"
    patchembed: Literal["v2"] = "v2"
    norm_layer: Literal["ln2d"] = "ln2d"
    use_v2d: bool = True
    image_size: int = 224
    num_classes: int = 2

    aux: dict[str, Any] = field(default_factory=dict)
    resolution: tuple[int, int] = (224, 224)

    def __post_init__(self):
        assert_check_literals(self)
        assert self.image_size == self.input_shape[0]
        assert self.num_classes == self.output_shape[0]
        assert self.resolution == self.input_shape[:2]


@register
class TwoDMamba(BaseModelTorch):
    config: TwoDMambaConfig

    def __init__(self, config: TwoDMambaConfig):
        super().__init__(config)
        self.config = config
        import os
        import sys

        sys.path.append(os.path.split(os.path.abspath(__file__))[0] + "/../../2DMamba/2DVMamba")
        from classification.models.vmamba import VSSM

        self.model = VSSM(
            dims=config.dims,
            depths=config.depths,
            ssm_d_state=config.ssm_d_state,
            ssm_dt_rank=config.ssm_dt_rank,
            ssm_ratio=config.ssm_ratio,
            ssm_conv_bias=config.ssm_conv_bias,
            ssm_forward_type=config.ssm_forward_type,
            mlp_ratio=config.mlp_ratio,
            downsample=config.downsample,
            patchembed=config.patchembed,
            norm_layer=config.norm_layer,
            use_v2d=config.use_v2d,
            num_classes=config.num_classes,
            image_size=config.image_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.permute(0, 3, 1, 2))


@dataclass
class EfficientNetConfig(ConfigInterface):
    _short_name: str = "EfficientNet"
    input_dim: int = 96
    input_shape: tuple[int, int, int] = (224, 224, 3)
    output_shape: tuple[int] = (2,)
    dtype: str = "float32"
    patch_size: int = 16

    aux: dict[str, Any] = field(default_factory=dict)
    resolution: tuple[int, int] = (224, 224)
    num_classes: int = 2

    def __post_init__(self):
        assert self.num_classes == self.output_shape[0]


@register
class EfficientNet(BaseModelTorch):
    config: EfficientNetConfig

    def __init__(self, config: EfficientNetConfig):
        super().__init__(config)
        self.config = config

        self.model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b0",
            trust_repo=True,
            num_classes=config.num_classes,
            pretrained=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.permute(0, 3, 1, 2))


### changed for torch
# PyTorch ViT implementation based on the JAX version
@dataclass
class ViTConfig(BaseModelTorchConfig):
    """Configuration for PyTorch Vision Transformer."""

    layers: int = 12
    dim: int = 768
    heads: int = 12
    labels: int | None = 1000
    layerscale: bool = False

    patch_size: int = 16
    image_size: int = 224
    posemb: Literal["learnable", "sincos2d"] = "learnable"
    pooling: Literal["cls", "gap"] = "cls"

    dropout: float = 0.0
    droppath: float = 0.0
    grad_ckpt: bool = False

    def __post_init__(self):
        # Set output_shape based on labels
        if self.labels is not None:
            self.output_shape = (self.labels,)
        # Set resolution based on image_size
        self.resolution = (self.image_size, self.image_size)

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2


def fixed_sincos2d_embeddings_torch(ncols: int, nrows: int, dim: int) -> torch.Tensor:
    """PyTorch version of fixed sincos 2D embeddings."""
    freqs = 1 / (10000 ** torch.linspace(0, 1, dim // 4))
    x = torch.outer(torch.arange(0, nrows, dtype=torch.float32), freqs)
    y = torch.outer(torch.arange(0, ncols, dtype=torch.float32), freqs)

    x = x[None, :, :].expand(ncols, nrows, dim // 4)
    y = y[:, None, :].expand(ncols, nrows, dim // 4)
    return torch.cat((torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)), dim=2)


class PatchEmbedTorch(nn.Module):
    """PyTorch patch embedding layer."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.patch_embed = nn.Conv2d(3, config.dim, kernel_size=config.patch_size, stride=config.patch_size, bias=True)

        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)

        if config.posemb == "learnable":
            num_patches = config.num_patches[0] * config.num_patches[1]
            if config.pooling == "cls":
                num_patches += 1  # Add one for cls token
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, config.dim) * 0.02)
        elif config.posemb == "sincos2d":
            pos_embed = fixed_sincos2d_embeddings_torch(*config.num_patches, config.dim)
            self.register_buffer("pos_embed", pos_embed.flatten(0, 1).unsqueeze(0))
        else:
            self.pos_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)

        # Add positional embedding
        if hasattr(self, "pos_embed") and self.pos_embed is not None:
            if self.config.pooling == "cls":
                # Add cls token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
            x = x + self.pos_embed
        elif self.config.pooling == "cls":
            # Add cls token without pos embed
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        return x


class AttentionTorch(nn.Module):
    """PyTorch multi-head attention using SDPA primitives."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.heads
        self.head_dim = config.head_dim

        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=True)
        self.proj = nn.Linear(config.dim, config.dim, bias=True)
        self.dropout_p = config.dropout

        # Initialize weights
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### changed for torch - using SDPA primitives
        # Use PyTorch's optimized scaled dot product attention
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=False)
        ###

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class FeedForwardTorch(nn.Module):
    """PyTorch feed-forward network."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.dim, config.hidden_dim, bias=True)
        self.fc2 = nn.Linear(config.hidden_dim, config.dim, bias=True)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ViTLayerTorch(nn.Module):
    """PyTorch ViT transformer layer."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = AttentionTorch(config)
        self.drop_path = DropPath(config.droppath)

        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = FeedForwardTorch(config)

        # Layer scale
        if config.layerscale:
            self.gamma1 = nn.Parameter(torch.ones(config.dim) * 1e-4)
            self.gamma2 = nn.Parameter(torch.ones(config.dim) * 1e-4)
        else:
            self.gamma1 = self.gamma2 = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


@register
class ViTTorchModel(BaseModelTorch):
    """PyTorch Vision Transformer model."""

    config: ViTConfig

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedTorch(config)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([ViTLayerTorch(config) for _ in range(config.layers)])

        self.norm = nn.LayerNorm(config.dim)

        # Classification head
        if config.labels is not None:
            self.head = nn.Linear(config.dim, config.labels, bias=True)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
        else:
            self.head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # If no classification head, return all tokens
        if self.head is None:
            return x

        # Pooling
        if self.config.pooling == "cls":
            x = x[:, 0]  # Use cls token
        elif self.config.pooling == "gap":
            x = x.mean(dim=1)  # Global average pooling

        x = self.head(x)
        return x


# Simple CNN for testing
@dataclass
class SimpleCNNConfig(BaseModelTorchConfig):
    """Configuration for a simple CNN model."""

    num_layers: int = 3
    hidden_dim: int = 64
    dropout: float = 0.1


class SimpleCNNModel(nn.Module):
    """Simple CNN PyTorch model implementation."""

    def __init__(self, config: SimpleCNNConfig):
        super().__init__()
        self.config = config

        # Simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, config.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(config.hidden_dim, config.hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(config.hidden_dim * 2, config.hidden_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.output_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x


@register
class SimpleCNN(BaseModelTorch):
    """Simple CNN model for testing purposes."""

    config: SimpleCNNConfig

    def __init__(self, config: SimpleCNNConfig):
        self.config = config

    def instantiate(self, interface_type) -> nn.Module:
        """Create the actual PyTorch model."""
        return SimpleCNNModel(self.config)


###
