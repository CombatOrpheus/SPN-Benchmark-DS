"""
Tests for the PyTorch Lightning implementation of the GNN module.
"""

import pytest
import torch
import pytorch_lightning as pl
from spn_datasets.gnns.lightning_data import SPNDataModule
from spn_datasets.gnns.lightning_models import GCNLightning, GATLightning, MPNNLightning

HDF5_PATH = "tests/temp_data/data_hdf5/test_data.h5"


@pytest.fixture(scope="module")
def data_module():
    """
    Provides a reusable SPNDataModule instance for the tests.
    """
    return SPNDataModule(hdf5_path=HDF5_PATH, graph_type="reachability", batch_size=4)


def test_data_module_instantiation(data_module):
    """
    Tests that the SPNDataModule can be instantiated.
    """
    assert data_module is not None


def test_data_module_setup(data_module):
    """
    Tests that the SPNDataModule setup runs correctly.
    """
    data_module.setup()
    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert data_module.test_dataset is not None

    # Check that the splits have the correct size
    num_samples = len(data_module.dataset)
    train_len = int(num_samples * 0.8)
    val_len = int(num_samples * 0.1)
    test_len = num_samples - train_len - val_len
    assert len(data_module.train_dataset) == train_len
    assert len(data_module.val_dataset) == val_len
    assert len(data_module.test_dataset) == test_len


def test_data_module_dataloaders(data_module):
    """
    Tests that the DataLoaders are created and return batches of the correct size.
    """
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Check that the loaders return batches
    train_batch = next(iter(train_loader))
    assert train_batch is not None
    assert train_batch.batch.max() == data_module.batch_size - 1


@pytest.mark.parametrize("model_class", [GCNLightning, GATLightning, MPNNLightning])
def test_lightning_model_instantiation(model_class):
    """
    Tests that the Lightning models can be instantiated.
    """
    in_channels = 10
    out_channels = 2
    model = model_class(in_channels=in_channels, out_channels=out_channels)
    assert model is not None


def test_trainer_fast_dev_run():
    """
    Tests that a fast dev run (a single batch) completes without errors.
    """
    data_module = SPNDataModule(hdf5_path=HDF5_PATH, graph_type="reachability", batch_size=4)
    data_module.setup()
    in_channels = data_module.dataset.num_node_features
    out_channels = data_module.dataset.num_classes

    model = GCNLightning(in_channels=in_channels, out_channels=out_channels)

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=data_module)
