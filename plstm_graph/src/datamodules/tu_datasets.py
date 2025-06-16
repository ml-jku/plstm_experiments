import pytorch_lightning as pl
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset


def get_splits(dataset, seed, fold_idx):
    try:
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        idx_list = []

        for idx in kfold.split(np.zeros(len(dataset.y)), dataset.y):
            idx_list.append(idx)

        train_val_idx, test_idx = idx_list[fold_idx]

        train_idx, val_idx = train_test_split(train_val_idx, train_size=0.888889, random_state=seed, stratify=dataset.y[train_val_idx])

    except ValueError:
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
        idx_list = []

        for idx in kfold.split(np.zeros(len(dataset.y)), dataset.y):
            idx_list.append(idx)

        train_val_idx, test_idx = idx_list[fold_idx]
        train_idx, val_idx = train_test_split(train_val_idx, train_size=0.888889, random_state=seed)

    return train_idx, val_idx, test_idx


def subset_from_indices(dataset, indices):
    return [dataset[i] for i in indices]


class Dataset(InMemoryDataset):
    def __init__(self, dataset_name, fold_idx, mode, use_node_attr=True, seed=42):
        dataset = TUDataset(f'./data/', dataset_name, use_node_attr=use_node_attr)
        train_idx, val_idx, test_idx = get_splits(dataset, seed=seed, fold_idx=fold_idx)

        # set node features to 1 for datasets that do not contain node features
        if 'x' not in dataset[0]:
            processed_dataset = []
            for i in range(len(dataset)):
                g = dataset[i]
                features = torch.ones((g.num_nodes, 1))
                g['x'] = features
                processed_dataset.append(g)

            dataset = processed_dataset

        if mode == 'train':
            dataset = subset_from_indices(dataset, train_idx)
        elif mode == 'test':
            dataset = subset_from_indices(dataset, test_idx)
        else:
            dataset = subset_from_indices(dataset, val_idx)

        super().__init__(f'./data/{dataset_name}')
        self.data, self.slices = self.collate(dataset)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader
    ):
        super().__init__()
        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader
        self.test_dataloader_ = test_dataloader

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_
    
    def test_dataloader(self):
        return self.test_dataloader_





