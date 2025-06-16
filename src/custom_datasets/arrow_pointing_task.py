from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.utils.data as data
from compoconf import ConfigInterface, register
import jax

from arrow_pointing_dataset import ArrowPointingConfig as APConfig, ArrowPointingTorchDataset, create_dataloader

from jax_trainer.datasets.collate import build_batch_collate, batch_collate
from jax_trainer.datasets.data_struct import DatasetConfig, DatasetModule, SupervisedBatch
from jax_trainer.datasets.utils import build_data_loaders
from jax_trainer.datasets.examples import LimitDataset
from .mix_collator import MixCollator
from .util import apply_override, deepcopy_with_dataclasses

# from .custom_transforms import ComposeTransform
import torchvision.transforms.v2 as T
from mill.vision.pretransforms import TorchVisionTransform


def batch_wrapper_fn(batch):
    return SupervisedBatch(batch[0].shape[0], np.array(batch[0]), np.array(batch[1]))


@dataclass
class ArrowPointingConfig(DatasetConfig, ConfigInterface):
    """Configuration for ArrowPointing dataset."""

    arrow_pointing_config: APConfig = field(default_factory=APConfig)
    transforms: list[TorchVisionTransform.cfgtype] = field(default_factory=list)
    num_classes: int = 2
    limit_train_size: Optional[int] = None
    train_mix_collator: Optional[Dict[str, Any]] = None
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


@dataclass
class ArrowPointingExtrapolationDatasetOldConfig(ArrowPointingConfig):
    pass


@dataclass
class ArrowPointingFastDatasetConfig(ArrowPointingConfig):
    pass


@register
class ArrowPointingDataset(DatasetModule):
    """ArrowPointing dataset implementation."""

    config: ArrowPointingConfig

    def __init__(self, config: ArrowPointingConfig, mesh: Optional[jax.sharding.Mesh] = None):
        """Initialize ArrowPointing dataset.

        Args:
            config: Configuration for the dataset.
            mesh: Optional mesh for distributed training.
        """
        self.config = config

        # Build transforms for each split

        # Loading the training/validation set
        pointing_config = deepcopy_with_dataclasses(config.arrow_pointing_config)
        pointing_config.seed = pointing_config.seed
        # pointing_config = APConfig(**pointing_config)
        train_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.train_transforms])
            if config.train_transforms
            else None
        )

        train_dataset = ArrowPointingTorchDataset(pointing_config, transform=train_transform)

        train_set, val_set = data.random_split(
            train_dataset,
            [len(train_dataset) - config.val_size, config.val_size],
            generator=torch.Generator().manual_seed(config.split_seed),
        )
        train_set = LimitDataset(train_set, limit=config.limit_train_size or len(train_set))

        # Loading the test set
        test_config = deepcopy_with_dataclasses(config)
        if config.test_overrides:
            apply_override(test_config, config.test_overrides)
        test_config.arrow_pointing_config.seed = test_config.arrow_pointing_config.seed + 1000000
        test_config.arrow_pointing_config.n_samples = config.test_size

        test_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.test_transforms])
            if config.train_transforms
            else None
        )
        test_set = ArrowPointingTorchDataset(test_config.arrow_pointing_config, transform=test_transform)

        train_mix_collator = config.train_mix_collator

        if train_mix_collator is not None:
            train_mix_collator = MixCollator(num_classes=config.num_classes, **train_mix_collator)

            def train_collate_fn(x):
                return batch_collate(SupervisedBatch, train_mix_collator(x))
        else:
            train_collate_fn = build_batch_collate(SupervisedBatch)

        train_loader, val_loader, test_loader = build_data_loaders(
            train_set,
            val_set,
            test_set,
            train=[True, False, False],
            collate_fn=[
                train_collate_fn,
                build_batch_collate(SupervisedBatch),
                build_batch_collate(SupervisedBatch),
            ],
            world_size=jax.process_count(),
            rank=jax.process_index(),
            mesh=mesh,
            config=config,
        )

        super().__init__(
            config=config,
            train=train_set,
            val=val_set,
            test=test_set,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )


@register
class ArrowPointingExtrapolationDatasetOld(DatasetModule):
    """ArrowPointing dataset implementation with separate validation and test configurations."""

    config: ArrowPointingExtrapolationDatasetOldConfig

    def __init__(self, config: ArrowPointingExtrapolationDatasetOldConfig, mesh: Optional[jax.sharding.Mesh] = None):
        """Initialize ArrowPointing dataset with extrapolation settings.

        Args:
            config: Configuration for the dataset.
            mesh: Optional mesh for distributed training.
        """
        self.config = config

        # Loading the training set
        pointing_config = deepcopy_with_dataclasses(config.arrow_pointing_config)
        train_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.train_transforms])
            if config.train_transforms
            else None
        )
        train_dataset = ArrowPointingTorchDataset(pointing_config, transform=train_transform)
        train_set = LimitDataset(train_dataset, limit=config.limit_train_size or len(train_dataset))

        # Loading the validation set
        val_config = deepcopy_with_dataclasses(config)
        if config.val_overrides:
            apply_override(val_config, config.val_overrides)
        val_config.arrow_pointing_config.seed = val_config.arrow_pointing_config.seed + 1000000
        val_config.arrow_pointing_config.n_samples = config.val_size

        val_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.val_transforms])
            if config.val_transforms
            else None
        )
        val_set = ArrowPointingTorchDataset(val_config.arrow_pointing_config, transform=val_transform)

        # Loading the test set
        test_config = deepcopy_with_dataclasses(config)
        if config.test_overrides:
            apply_override(test_config, config.test_overrides)
        test_config.arrow_pointing_config.seed = test_config.arrow_pointing_config.seed + 2000000
        test_config.arrow_pointing_config.n_samples = config.test_size

        test_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.test_transforms])
            if config.test_transforms
            else None
        )
        test_set = ArrowPointingTorchDataset(test_config.arrow_pointing_config, transform=test_transform)

        train_mix_collator = config.train_mix_collator

        if train_mix_collator is not None:
            train_mix_collator = MixCollator(num_classes=config.num_classes, **train_mix_collator)

            def train_collate_fn(x):
                return batch_collate(SupervisedBatch, train_mix_collator(x))
        else:
            train_collate_fn = build_batch_collate(SupervisedBatch)

        train_loader, val_loader, test_loader = build_data_loaders(
            train_set,
            val_set,
            test_set,
            train=[True, False, False],
            collate_fn=[
                train_collate_fn,
                build_batch_collate(SupervisedBatch),
                build_batch_collate(SupervisedBatch),
            ],
            world_size=jax.process_count(),
            rank=jax.process_index(),
            mesh=mesh,
            config=config,
        )

        super().__init__(
            config=config,
            train=train_set,
            val=val_set,
            test=test_set,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )


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


@register
class ArrowPointingExtrapolationDataset(DatasetModule):
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

        train_collate_fn = build_batch_collate(SupervisedBatch)

        train_loader = build_data_loaders(
            train_set,
            train=[True],
            collate_fn=[
                train_collate_fn,
                build_batch_collate(SupervisedBatch),
                build_batch_collate(SupervisedBatch),
            ],
            world_size=jax.process_count(),
            rank=jax.process_index(),
            mesh=mesh,
            config=config,
        )[0]

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

                val_loader[val_subset] = build_data_loaders(
                    val_set[val_subset],
                    train=[False],
                    collate_fn=[
                        build_batch_collate(SupervisedBatch),
                    ],
                    world_size=jax.process_count(),
                    rank=jax.process_index(),
                    mesh=mesh,
                    config=config,
                )[0]

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
                test_loader[test_subset] = build_data_loaders(
                    test_set[test_subset],
                    train=[False],
                    collate_fn=[
                        build_batch_collate(SupervisedBatch),
                    ],
                    world_size=jax.process_count(),
                    rank=jax.process_index(),
                    mesh=mesh,
                    config=config,
                )[0]

        super().__init__(
            config=config,
            train=train_set,
            val=val_set,
            test=test_set,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )


@register
class ArrowPointingFastDataset(DatasetModule):
    """ArrowPointing dataset implementation with direct dataloader creation."""

    config: ArrowPointingFastDatasetConfig

    def __init__(self, config: ArrowPointingFastDatasetConfig, mesh: Optional[jax.sharding.Mesh] = None):
        """Initialize ArrowPointing dataset with fast dataloader creation.

        Args:
            config: Configuration for the dataset.
            mesh: Optional mesh for distributed training.
        """
        self.config = config

        # Loading the training set
        pointing_config = deepcopy_with_dataclasses(config.arrow_pointing_config)
        pointing_config.seed = pointing_config.seed

        train_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.train_transforms])
            if config.train_transforms
            else None
        )
        train_set = ArrowPointingTorchDataset(pointing_config, transform=train_transform)

        train_loader = create_dataloader(
            pointing_config,
            batch_size=config.local_batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            batch_wrapper_fn=batch_wrapper_fn,
        )

        # Loading the validation set
        pointing_config = deepcopy_with_dataclasses(config.arrow_pointing_config)
        pointing_config.seed = pointing_config.seed + 1000000
        pointing_config.n_samples = config.val_size

        val_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.val_transforms])
            if config.train_transforms
            else None
        )
        val_set = ArrowPointingTorchDataset(pointing_config, transform=val_transform)
        val_loader = create_dataloader(
            pointing_config,
            batch_size=config.local_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            batch_wrapper_fn=batch_wrapper_fn,
        )

        # Loading the test set
        # Loading the test set
        test_config = deepcopy_with_dataclasses(config)
        if config.test_overrides:
            apply_override(test_config, config.test_overrides)
        test_config.arrow_pointing_config.seed = test_config.arrow_pointing_config.seed + 1000000
        test_config.arrow_pointing_config.n_samples = config.test_size

        test_transform = (
            T.Compose([t.instantiate(TorchVisionTransform) for t in config.test_transforms])
            if config.test_transforms
            else None
        )
        test_set = ArrowPointingTorchDataset(test_config.arrow_pointing_config, transform=test_transform)

        test_loader = create_dataloader(
            pointing_config,
            batch_size=config.local_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            batch_wrapper_fn=batch_wrapper_fn,
        )

        super().__init__(
            config=config,
            train=train_set,
            val=val_set,
            test=test_set,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )


# Legacy functions for backward compatibility
def build_arrowpointing_datasets_fast(dataset_config, mesh=None):
    """Legacy function for backward compatibility.

    Args:
        dataset_config: Configuration for the dataset.
        mesh: Optional mesh for distributed training.

    Returns:
        DatasetModule object.
    """
    config = ArrowPointingConfig(
        arrow_pointing_config=dataset_config.arrow_pointing_config,
        train_transforms=dataset_config.get("train_transforms", []),
        val_transforms=dataset_config.get("val_transforms", []),
        test_transforms=dataset_config.get("test_transforms", []),
        val_size=dataset_config.get("val_size", 5120),
        test_size=dataset_config.get("test_size", 5120),
        num_workers=dataset_config.get("num_workers", 0),
        local_batch_size=dataset_config.get("local_batch_size", 128),
        pin_memory=dataset_config.get("pin_memory", True),
        prefetch_factor=dataset_config.get("prefetch_factor", 4),
    )

    return ArrowPointingFastDataset(config, mesh)


def build_arrowpointing_datasets(dataset_config, mesh=None):
    """Legacy function for backward compatibility.

    Args:
        dataset_config: Configuration for the dataset.
        mesh: Optional mesh for distributed training.

    Returns:
        DatasetModule object.
    """
    config = ArrowPointingConfig(
        arrow_pointing_config=dataset_config.arrow_pointing_config,
        train_transforms=dataset_config.get("train_transforms", []),
        val_transforms=dataset_config.get("val_transforms", []),
        test_transforms=dataset_config.get("test_transforms", []),
        val_size=dataset_config.get("val_size", 5120),
        test_size=dataset_config.get("test_size", 10240),
        split_seed=dataset_config.get("split_seed", 42),
        num_classes=dataset_config.get("num_classes", 2),
        limit_train_size=dataset_config.get("limit_train_size", None),
        train_mix_collator=dataset_config.get("train_mix_collator", None),
        test_overrides=getattr(dataset_config, "test_overrides", None),
        pin_memory=dataset_config.get("pin_memory", True),
        prefetch_factor=dataset_config.get("prefetch_factor", 4),
    )

    return ArrowPointingDataset(config, mesh)


def build_arrowpointing_datasets_extrapolation(dataset_config, mesh=None):
    """Legacy function for backward compatibility.

    Args:
        dataset_config: Configuration for the dataset.
        mesh: Optional mesh for distributed training.

    Returns:
        DatasetModule object.
    """
    config = ArrowPointingConfig(
        arrow_pointing_config=dataset_config.arrow_pointing_config,
        train_transforms=dataset_config.get("train_transforms", []),
        val_transforms=dataset_config.get("val_transforms", []),
        test_transforms=dataset_config.get("test_transforms", []),
        val_size=dataset_config.get("val_size", 5120),
        test_size=dataset_config.get("test_size", 10240),
        split_seed=dataset_config.get("split_seed", 42),
        num_classes=dataset_config.get("num_classes", 2),
        limit_train_size=dataset_config.get("limit_train_size", None),
        train_mix_collator=dataset_config.get("train_mix_collator", None),
        test_overrides=getattr(dataset_config, "test_overrides", None),
        val_overrides=getattr(dataset_config, "val_overrides", None),
        pin_memory=dataset_config.get("pin_memory", True),
        prefetch_factor=dataset_config.get("prefetch_factor", 4),
    )

    return ArrowPointingExtrapolationDataset(config, mesh)
