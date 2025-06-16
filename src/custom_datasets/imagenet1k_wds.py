from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import jax
import torch.utils.data as data
from compoconf import ConfigInterface, register
from jax_trainer.datasets.data_struct import DatasetConfig, DatasetModule, SupervisedBatch
from jax_trainer.datasets.collate import build_batch_collate, batch_collate, numpy_collate
from jax_trainer.datasets.multihost_dataloading import MultiHostDataLoadIterator
from collections.abc import Sequence, Callable, Iterator
from torch.utils.data import DataLoader, default_collate
import numpy as np

from .mix_collator import MixCollator
import webdataset as wds
import os
import itertools
import copy
from typing import Any, Tuple
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


IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])


def auto_augment_factory(args) -> T.Transform:
    aa_hparams = {
        "translate_const": int(args.image_size * 0.45),
        "img_mean": tuple((IMAGENET_DEFAULT_MEAN * 0xFF).astype(int)),
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

        data_pipeline = (
            (wds.SimpleShardList(sorted([str(data_path / fname) for fname in os.listdir(data_path)]), seed),)
            + (
                ()
                if split != "train"
                else (
                    itertools.cycle,
                    wds.detshuffle(),
                )
            )
            + (
                wds.slice(jax.process_index(), None, jax.process_count()),
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            )
            + (() if split != "train" else (wds.detshuffle(),))
            + (
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.to_tuple("jpg", "cls", handler=wds.ignore_and_continue),
                partial(repeat_samples, repeats=augment_repeats),
                wds.map_tuple(transform, partial(torch.tensor, device="cpu")),
            )
        )
        self.dataset = wds.DataPipeline(*data_pipeline)

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
    prefetch_factor: int = 4
    shuffle_seed: int = 42
    channels_first: bool = False
    _short_name: str = ""
    aux: dict[str, Any] = field(default_factory=dict)


@register
class ImageNet1kWDSDataset(DatasetModule):
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

        # Setup collation functions
        train_mix_collator = config.train_mix_collator

        if train_mix_collator is not None:
            train_mix_collator = MixCollator(num_classes=config.num_classes, **train_mix_collator)

            def train_collate_fn(x):
                return batch_collate(SupervisedBatch, jax.tree.map(np.array, train_mix_collator(x)))
        else:

            def train_collate_fn(x):
                return batch_collate(SupervisedBatch, jax.tree.map(np.array, default_collate(x)))

        def val_collate_fn(x):
            return batch_collate(SupervisedBatch, jax.tree.map(np.array, default_collate(x)))

        # Create data loaders
        train_loader = DataLoader(
            train_set,
            batch_size=config.global_batch_size // jax.process_count(),
            num_workers=config.num_workers,
            collate_fn=train_collate_fn,
            drop_last=True,
            prefetch_factor=20 if config.num_workers > 0 else None,
            persistent_workers=True if config.num_workers > 0 else False,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=config.global_batch_size // jax.process_count(),
            num_workers=config.num_workers,
            collate_fn=val_collate_fn,
            drop_last=True,
            prefetch_factor=20 if config.num_workers > 0 else None,
            persistent_workers=True if config.num_workers > 0 else False,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=config.global_batch_size // jax.process_count(),
            num_workers=config.num_workers,
            collate_fn=val_collate_fn,
            drop_last=True,
            prefetch_factor=20 if config.num_workers > 0 else None,
            persistent_workers=True if config.num_workers > 0 else False,
        )

        # Apply mesh if provided
        if mesh is not None:
            train_loader, val_loader, test_loader = map(
                lambda loader, dataset: MultiHostDataLoadIterator(
                    loader,
                    iterator_length=len(dataset) // config.global_batch_size,
                    global_mesh=mesh,
                    reset_after_epoch=True,
                ),
                (train_loader, val_loader, test_loader),
                (train_set, val_set, test_set),
            )
        # print(
        #     "DATASET LENGTH:",
        #     len(train_set) // config.global_batch_size,
        #     "LOADER LENGTH",
        #     len(train_loader),
        #     len(train_set) // config.global_batch_size,
        # )

        super().__init__(
            config=config,
            train=train_set,
            val=val_set,
            test=test_set,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )


# Legacy function for backward compatibility
def build_imagenet1kwds_datasets(dataset_config, mesh=None):
    """Legacy function for backward compatibility.

    Args:
        dataset_config: Configuration for the dataset.
        mesh: Optional mesh for distributed training.

    Returns:
        DatasetModule object.
    """
    config = ImageNet1kWDSConfig(
        shard_dir=getattr(dataset_config, "shard_dir"),
        resolution=getattr(dataset_config, "resolution", (224, 224)),
        color_jitter=getattr(dataset_config, "color_jitter", 0.3),
        random_erasing=getattr(dataset_config, "random_erasing", 0.0),
        test_crop_ratio=getattr(dataset_config, "test_crop_ratio", 1.0),
        auto_augment=getattr(dataset_config, "auto_augment", "3a"),
        random_crop=getattr(dataset_config, "random_crop", "rrc"),
        augment_repeats=getattr(dataset_config, "augment_repeats", 1),
        num_classes=dataset_config.get("num_classes", 1000),
        train_mix_collator=dataset_config.get("train_mix_collator", None),
        num_workers=dataset_config.get("num_workers", 4),
        global_batch_size=getattr(dataset_config, "global_batch_size", 128),
        # batch_size=dataset_config.get("batch_size", 128),
        pin_memory=dataset_config.get("pin_memory", True),
        prefetch_factor=dataset_config.get("prefetch_factor", 4),
    )

    return ImageNet1kWDSDataset(config, mesh)
