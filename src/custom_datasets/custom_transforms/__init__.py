from .cifar100_norm import Cifar100Norm
from .cifar10_norm import Cifar10Norm
from .imagenet1k_norm import Imagenet1kNorm
from .color_jitter import ColorJitter
from .compose_transform import ComposeTransform
from .gaussian_blur_pil import GaussianBlur, GaussianBlurPIL
from .image_moment_norm import ImageMomentNorm
from .image_range_norm import ImageRangeNorm
from .norm_base import NormBase
from .rand_augment import RandAugment
from .random_resized_crop import RandomResizedCrop
from .resize_transform import Resize
from .segmentation_pad import SegmentationPad
from .segmentation_random_crop import SegmentationRandomCrop
from .segmentation_random_horizontal_flip import SegmentationRandomHorizontalFlip
from .segmentation_random_resize import SegmentationRandomResize
from .stochastic_transform import StochasticTransform
from .three_augment import ThreeAugment
from .transform import Transform
from .master_factory import MasterFactory
from .transform_factory import TransformFactory
from .one_hot_label_transform import OneHotLabelTransform

MasterFactory.set(
    "transform",
    TransformFactory(
        module_names=[
            "custom_transforms",
            "torchvision.transforms",
        ],
    ),
)


__all__ = [
    "Cifar100Norm",
    "Cifar10Norm",
    "Imagenet1kNorm",
    "ColorJitter",
    "ComposeTransform",
    "GaussianBlur",
    "GaussianBlurPIL",
    "ImageMomentNorm",
    "ImageRangeNorm",
    "Imagenet1kNorm",
    "NormBase",
    "RandAugment",
    "RandomResizedCrop",
    "Resize",
    "SegmentationPad",
    "SegmentationRandomCrop",
    "SegmentationRandomHorizontalFlip",
    "SegmentationRandomResize",
    "StochasticTransform",
    "ThreeAugment",
    "Transform",
    "OneHotLabelTransform",
]
