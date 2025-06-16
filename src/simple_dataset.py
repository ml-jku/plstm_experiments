from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import jax
import torch.utils.data as data
from compoconf import ConfigInterface, register, RegistrableConfigInterface, register_interface
from jax_trainer.datasets.data_struct import (
    DatasetConfig,
)
from collections.abc import Iterator
from torch.utils.data import DataLoader, default_collate
import numpy as np

import webdataset as wds
import os
import itertools
import copy
from typing import Tuple
from functools import partial
import torch
from pathlib import Path
import torchvision.transforms.v2 as T
from torch import nn
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
import argparse


from arrow_pointing_dataset import (
    ArrowPointingConfig as APConfig,
    ArrowPointingTorchDataset,
)

from jax_trainer.datasets.examples import LimitDataset
from .utils import apply_override, deepcopy_with_dataclasses

# from .custom_transforms import ComposeTransform
from mill.vision.pretransforms import TorchVisionTransform


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


@register_interface
class SimpleDatasetModule(RegistrableConfigInterface):
    train_dataset: Any
    val_dataset: Any
    train_dataloader: DataLoader
    val_dataloader: DataLoader | dict[str, DataLoader]
    test_dataset: Any | None = None
    test_dataloader: DataLoader | dict[str, DataLoader] | None


def auto_augment_factory(args) -> T.Transform:
    aa_hparams = {
        "translate_const": int(args.image_size * 0.45),
        "img_mean": tuple((np.array(IMAGENET_DEFAULT_MEAN) * 0xFF).astype(int)),
    }
    if args.auto_augment == "none":
        return T.Identity()
    if args.auto_augment.startswith("rand"):
        return rand_augment_transform(args.auto_augment, aa_hparams)
    if args.auto_augment.startswith("augmix"):
        aa_hparams["translate_pct"] = 0.3
        return augment_and_mix_transform(args.auto_augment, aa_hparams)
    return auto_augment_transform(args.auto_augment, aa_hparams)


def create_transforms(args) -> tuple[nn.Module, nn.Module]:
    if args.random_crop == "rrc":
        train_transforms = [T.RandomResizedCrop(args.image_size, interpolation=3)]
    elif args.random_crop == "src":
        train_transforms = [
            T.Resize(args.image_size, interpolation=3),
            T.RandomCrop(args.image_size, padding=4, padding_mode="reflect"),
        ]
    elif args.random_crop == "none":
        train_transforms = [
            T.Resize(args.image_size, interpolation=3),
            T.CenterCrop(args.image_size),
        ]

    train_transforms += [
        T.RandomHorizontalFlip(),
        auto_augment_factory(args),
        T.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter),
        T.RandomErasing(args.random_erasing, value="random"),
        T.PILToTensor(),
    ]
    valid_transforms = [
        T.Resize(int(args.image_size / args.test_crop_ratio), interpolation=3),
        T.CenterCrop(args.image_size),
        T.PILToTensor(),
    ]
    return T.Compose(train_transforms), T.Compose(valid_transforms)


def repeat_samples(samples: Iterator[Any], repeats: int = 1) -> Iterator[Any]:
    for sample in samples:
        for _ in range(repeats):
            yield copy.deepcopy(sample)


class ImageNetDataset(data.IterableDataset):
    def __init__(self, shard_dir, split: str, seed: int = 42, augment_repeats=1, transform=None, **kwargs):
        data_path = Path(shard_dir) / split
        self.split = split
        self.data_path = data_path

        if split == "train":
            dataset = wds.DataPipeline(
                wds.SimpleShardList(str(data_path) + "/imagenet1k-train-{0000..1023}.tar", seed=seed),
                itertools.cycle,
                wds.detshuffle(),
                wds.slice(jax.process_index(), None, jax.process_count()),
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.ignore_and_continue),
                wds.detshuffle(),
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.to_tuple("jpg", "cls", handler=wds.ignore_and_continue),
                partial(repeat_samples, repeats=augment_repeats),
                wds.map_tuple(transform, partial(torch.tensor, device="cpu")),
            )
        else:
            dataset = wds.DataPipeline(
                # wds.SimpleShardList(sorted([str(data_path / fname) for fname in os.listdir(data_path)])),
                wds.SimpleShardList(str(data_path) + "/imagenet1k-validation-{00..63}.tar"),
                wds.slice(jax.process_index(), None, jax.process_count()),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode("pil"),
                wds.to_tuple("jpg", "cls"),
                wds.map_tuple(transform, partial(torch.tensor, device="cpu")),
            )

        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        if self.split == "train":
            # return 240219
            return 1281167
        if self.split == "val":
            return 50000
        if self.split == "test":
            return 100000
        raise ValueError


def collate_and_shuffle(batch: list[Any], repeats: int = 1) -> Any:
    return default_collate(sum([batch[i::repeats] for i in range(repeats)], []))


def collate_and_pad(batch: list[Any], batch_size: int = 1) -> Any:
    pad = tuple(torch.full_like(x, fill_value=-1) for x in batch[0])
    return default_collate(batch + [pad] * (batch_size - len(batch)))


@dataclass
class ImageNet1kWDSConfig(DatasetConfig, ConfigInterface):
    """Configuration for ImageNet1k WebDataset."""

    shard_dir: str = ""
    resolution: Tuple[int, int] = field(default_factory=lambda: (224, 224))
    color_jitter: float = 0.3
    random_erasing: float = 0.0
    test_crop_ratio: float = 1.0
    auto_augment: str = "3a"  # "rand-m9-mstd0.5-inc1"
    random_crop: str = "rrc"
    augment_repeats: int = 1
    num_classes: int = 1000
    num_channels: int = 3
    train_mix_collator: Optional[Dict[str, Any]] = None
    num_workers: int = 4
    global_batch_size: int = 128
    pin_memory: bool = True
    prefetch_factor: int = 20
    shuffle_seed: int = 42
    channels_first: bool = True
    _short_name: str = ""
    aux: dict[str, Any] = field(default_factory=dict)


@register
class ImageNet1kWDSDataset(SimpleDatasetModule):
    """ImageNet1k WebDataset implementation."""

    config: ImageNet1kWDSConfig

    def __init__(self, config: ImageNet1kWDSConfig, mesh: Optional[jax.sharding.Mesh] = None):
        """Initialize ImageNet1k WebDataset.

        Args:
            config: Configuration for the dataset.
            mesh: Optional mesh for distributed training.
        """
        self.config = config

        # Create transforms
        train_transform, val_transform = create_transforms(
            argparse.Namespace(
                image_size=config.resolution[0],
                color_jitter=config.color_jitter,
                random_erasing=config.random_erasing,
                test_crop_ratio=config.test_crop_ratio,
                auto_augment=config.auto_augment,
                random_crop=config.random_crop,
            )
        )
        test_transform = val_transform

        # Loading the training set
        train_set = ImageNetDataset(
            shard_dir=config.shard_dir,
            split="train",
            seed=config.shuffle_seed,
            transform=train_transform,
            augment_repeats=config.augment_repeats,
        )

        # Loading the validation and test sets
        val_set = ImageNetDataset(shard_dir=config.shard_dir, split="val", transform=val_transform)
        test_set = ImageNetDataset(shard_dir=config.shard_dir, split="test", transform=test_transform)

        # Create data loaders
        train_loader = DataLoader(
            train_set,
            batch_size=config.global_batch_size // jax.process_count(),
            num_workers=config.num_workers,
            collate_fn=partial(collate_and_shuffle, repeats=config.augment_repeats),
            drop_last=True,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            persistent_workers=True if config.num_workers > 0 else False,
        )

        val_loader, test_loader = map(
            lambda ds: DataLoader(
                ds,
                batch_size=(batch_size := config.global_batch_size // jax.process_count()),
                num_workers=config.num_workers,
                collate_fn=partial(collate_and_pad, batch_size=batch_size),
                drop_last=False,
                prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
                persistent_workers=True if config.num_workers > 0 else False,
            ),
            (val_set, test_set),
        )

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


@dataclass
class ArrowPointingExtrapolationDatasetConfig(DatasetConfig, ConfigInterface):
    """Configuration for ArrowPointing dataset."""

    arrow_pointing_config: APConfig = field(default_factory=APConfig)
    transforms: list[TorchVisionTransform.cfgtype] = field(default_factory=list)
    split_seed: int = 42
    num_classes: int = 2
    limit_train_size: Optional[int] = None
    test_overrides: Optional[Dict[str, Any]] = None
    val_overrides: Optional[Dict[str, Any]] = None
    num_workers: int = 0
    local_batch_size: int = 128
    global_batch_size: int = 128
    pin_memory: bool = True
    prefetch_factor: int = 4
    channels_first: bool = False
    _short_name: str = ""
    aux: dict[str, Any] = field(default_factory=dict)


class RandomIndexDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, seed=0, shuffle: bool = False, infinite: bool = False):
        self.dataset = dataset
        self.seed = seed
        self.shuffle = shuffle
        self.infinite = infinite

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            iter_start = 0
            iter_end = len(self.dataset)
        else:  # In a worker process
            # Split workload
            per_worker = int((len(self.dataset) - 1) // worker_info.num_workers + 1)
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))

        # Create an iterator over the specified range
        indices = list(range(iter_start, iter_end))
        if self.shuffle:
            np.random.seed(self.seed + iter_start)  # Ensure deterministic shuffling per worker
            np.random.shuffle(indices)
        for idx in itertools.cycle(indices) if self.infinite else indices:  # Cycle through the data indefinitely
            data, label = self.dataset[idx]
            yield data, label


@register
class ArrowPointingExtrapolationDataset(SimpleDatasetModule):
    """ArrowPointing dataset implementation with separate validation and test configurations."""

    config: ArrowPointingExtrapolationDatasetConfig

    def __init__(
        self,
        config: ArrowPointingExtrapolationDatasetConfig,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        """Initialize ArrowPointing dataset with extrapolation settings.

        Args:
            config: Configuration for the dataset.
            mesh: Optional mesh for distributed training.
        """
        self.config = config

        # Loading the training set
        pointing_config = deepcopy_with_dataclasses(config.arrow_pointing_config)
        train_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.transforms]) if config.transforms else None
        )
        train_dataset = ArrowPointingTorchDataset(pointing_config, transform=train_transform)
        train_set = LimitDataset(train_dataset, limit=config.limit_train_size or len(train_dataset))

        assert config.global_batch_size // jax.process_count() // len(jax.local_devices()) == config.local_batch_size

        train_loader = DataLoader(
            RandomIndexDataset(train_set, shuffle=True, infinite=True),
            batch_size=config.global_batch_size // jax.process_count(),
            num_workers=config.num_workers,
            collate_fn=partial(collate_and_shuffle, repeats=1),
            drop_last=True,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            persistent_workers=True,
        )

        # Loading the validation set
        val_set, val_loader = {}, {}
        if config.val_overrides:
            for val_subset, val_subconfig in config.val_overrides.items():
                val_config = deepcopy_with_dataclasses(config)
                apply_override(val_config, val_subconfig)

                val_transform = (
                    T.Compose([t.instantiate(TorchVisionTransform) for t in val_config.transforms])
                    if val_config.transforms
                    else None
                )
                val_set[val_subset] = ArrowPointingTorchDataset(
                    val_config.arrow_pointing_config, transform=val_transform
                )

                val_loader[val_subset] = DataLoader(
                    RandomIndexDataset(val_set[val_subset]),
                    batch_size=val_config.global_batch_size // jax.process_count(),
                    num_workers=val_config.num_workers,
                    collate_fn=partial(collate_and_pad, batch_size=val_config.global_batch_size),
                    drop_last=False,
                    prefetch_factor=val_config.prefetch_factor if val_config.num_workers > 0 else None,
                    persistent_workers=True,
                )

        # Loading the test set
        test_set, test_loader = {}, {}
        if config.test_overrides:
            for test_subset, test_subconfig in config.test_overrides.items():
                test_config = deepcopy_with_dataclasses(config)
                apply_override(test_config, test_subconfig)

                test_transform = (
                    T.Compose([t.instantiate(TorchVisionTransform) for t in test_config.transforms])
                    if test_config.transforms
                    else None
                )
                test_set[test_subset] = ArrowPointingTorchDataset(
                    test_config.arrow_pointing_config, transform=test_transform
                )
                test_loader[test_subset] = DataLoader(
                    RandomIndexDataset(test_set[test_subset]),
                    batch_size=test_config.global_batch_size // jax.process_count(),
                    num_workers=test_config.num_workers,
                    collate_fn=partial(collate_and_pad, batch_size=test_config.global_batch_size),
                    drop_last=False,
                    prefetch_factor=test_config.prefetch_factor if test_config.num_workers > 0 else None,
                    persistent_workers=True,
                )

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


@dataclass
class CIFAR10DatasetConfig(ConfigInterface):
    data_dir: str = ""


@register
class CIFAR10Dataset(SimpleDatasetModule):
    config: CIFAR10DatasetConfig

    def __init__(self, config: CIFAR10DatasetConfig):
        self.config = config
