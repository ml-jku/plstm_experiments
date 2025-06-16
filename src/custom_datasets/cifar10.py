import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from ml_collections import ConfigDict
from torchvision.datasets import CIFAR10, MNIST

from jax_trainer.datasets.collate import build_batch_collate, batch_collate
from jax_trainer.datasets.data_struct import DatasetModule, SupervisedBatch
from jax_trainer.datasets.transforms import image_to_numpy, normalize_transform
from jax_trainer.datasets.utils import build_data_loaders
import jax
from custom_transforms import ComposeTransform
from jax_trainer.datasets.examples import LimitDataset
from .mix_collator import MixCollator


def build_cifar10aug_datasets(dataset_config: ConfigDict, mesh: jax.sharding.Mesh | None = None):
    """Builds CIFAR10 datasets.

    Args:
        dataset_config: Configuration for the dataset.

    Returns:
        DatasetModule object.
    """
    # Build transforms for each split
    train_transform = ComposeTransform(transforms=dataset_config.get("train_transforms", []))
    print("TRAIN TRANSFORM", train_transform)
    val_transform = ComposeTransform(transforms=dataset_config.get("val_transforms", []))
    test_transform = ComposeTransform(transforms=dataset_config.get("test_transforms", []))

    # Loading the training/validation set
    train_dataset = CIFAR10(root=dataset_config.data_dir, train=True, transform=train_transform, download=True)
    val_size = dataset_config.get("val_size", 5000)
    split_seed = dataset_config.get("split_seed", 42)
    num_classes = dataset_config.get("num_classes", 10)
    train_set, val_set = data.random_split(
        train_dataset,
        [50000 - val_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    train_set = LimitDataset(train_set, limit=dataset_config.get("limit_train_size", len(train_set)))

    # Loading the test set
    test_set = CIFAR10(root=dataset_config.data_dir, train=False, transform=test_transform, download=True)

    train_mix_collator = dataset_config.get("train_mix_collator", None)

    if train_mix_collator is not None:
        train_mix_collator = MixCollator(num_classes=num_classes, **train_mix_collator)

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
        config=dataset_config,
    )

    return DatasetModule(dataset_config, train_set, val_set, test_set, train_loader, val_loader, test_loader)
