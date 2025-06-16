import numpy as np
from ml_collections import ConfigDict
import jax
import torch.utils.data as data
from datasets import load_dataset
from jax_trainer.datasets.data_struct import DatasetModule, SupervisedBatch
from jax_trainer.datasets.collate import build_batch_collate, batch_collate
from jax_trainer.datasets.utils import build_data_loaders

from custom_transforms import ComposeTransform
from jax_trainer.datasets.examples import LimitDataset
from .mix_collator import MixCollator


class ImageNetDataset(data.Dataset):
    def __init__(self, split, transform=None, **kwargs):
        self.dataset = load_dataset("ILSVRC/imagenet-1k", split=split, **kwargs)
        self.transform = transform
        self.split = split

    def __getitem__(self, idx):
        sample = next(iter(self.dataset.skip(idx).take(1)))
        image = sample["image"].convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        if self.split == "train":
            return 1281167
        if self.split == "validation":
            return 50000
        if self.split == "test":
            return 100000
        raise ValueError


def build_imagenet1k_datasets(dataset_config: ConfigDict, mesh: jax.sharding.Mesh | None = None):
    """Builds CIFAR10 datasets.

    Args:
        dataset_config: Configuration for the dataset.

    Returns:
        DatasetModule object.
    """
    # Build transforms for each split
    num_classes = dataset_config.get("num_classes", 1000)

    train_transform = ComposeTransform(transforms=dataset_config.get("train_transforms", []))
    val_transform = ComposeTransform(transforms=dataset_config.get("val_transforms", []))
    test_transform = ComposeTransform(transforms=dataset_config.get("test_transforms", []))

    # Loading the training/validation set
    train_dataset = ImageNetDataset(split="train", transform=train_transform)
    train_set = LimitDataset(train_dataset, limit=dataset_config.get("limit_train_size", len(train_dataset)))

    # Loading the test set
    val_set = ImageNetDataset(split="validation", transform=val_transform)
    test_set = ImageNetDataset(split="test", transform=test_transform)

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
