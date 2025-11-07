"""
This module provides a PyTorch Lightning DataModule for the SPN dataset.
"""

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from spn_datasets.gnns.data import SPNDataset


class SPNDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the SPN dataset.

    This DataModule handles loading the data, creating train/validation/test splits,
    and providing DataLoaders for each.

    Args:
        hdf5_path (str): The path to the HDF5 file.
        graph_type (str): The type of graph to create ('reachability' or 'petri_net').
        batch_size (int): The batch size for the DataLoaders.
        train_val_test_split (tuple): A tuple with the proportions for train, validation, and test splits.
    """

    def __init__(self, hdf5_path: str, graph_type: str = "reachability", batch_size: int = 32, train_val_test_split: tuple = (0.8, 0.1, 0.1)):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """
        Loads the full dataset and creates the train, validation, and test splits.
        """
        if self.dataset is None:
            self.dataset = SPNDataset(hdf5_path=self.hdf5_path, graph_type=self.graph_type)

            num_samples = len(self.dataset)
            num_train = int(num_samples * self.train_val_test_split[0])
            num_val = int(num_samples * self.train_val_test_split[1])
            num_test = num_samples - num_train - num_val

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, [num_train, num_val, num_test]
            )

    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test set.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
