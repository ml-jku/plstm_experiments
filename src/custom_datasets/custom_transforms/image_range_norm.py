import torch
import torchvision.transforms.functional as F

from .norm_base import NormBase


class ImageRangeNorm(NormBase):
    def normalize(self, x, inplace=True):
        if not torch.is_tensor(x):
            x = F.to_tensor(x)
        n_channels = x.size(0)
        values = tuple(0.5 for _ in range(n_channels))
        return F.normalize(x, mean=values, std=values, inplace=inplace)

    def denormalize(self, x, inplace=True):
        if not torch.is_tensor(x):
            x = F.to_tensor(x)
        n_channels = x.size(0)
        std = tuple(2.0 for _ in range(n_channels))
        mean = tuple(-0.5 for _ in range(n_channels))
        zero = tuple(0.0 for _ in range(n_channels))
        one = tuple(1.0 for _ in range(n_channels))
        x = F.normalize(x, mean=zero, std=std, inplace=inplace)
        return F.normalize(x, mean=mean, std=one, inplace=inplace)
