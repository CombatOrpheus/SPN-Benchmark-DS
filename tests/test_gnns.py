"""
Tests for the GNN module, including data loading, model instantiation,
and training.
"""

import pytest
import torch
from spn_datasets.gnns.data import SPNDataset
from spn_datasets.gnns.models import GCN, GAT, MPNN
from spn_datasets.gnns.train import train_and_evaluate


HDF5_PATH = "tests/temp_data/data_hdf5/test_data.h5"


@pytest.fixture(scope="module")
def dataset():
    """
    Provides a reusable SPNDataset instance for the tests.
    """
    return SPNDataset(hdf5_path=HDF5_PATH, graph_type="reachability")


def test_dataset_instantiation(dataset):
    """
    Tests that the SPNDataset can be instantiated and has the correct length.
    """
    assert dataset is not None
    assert len(dataset) == 10


def test_model_instantiation():
    """
    Tests that the GNN models can be instantiated with the correct dimensions.
    """
    in_channels = 10
    out_channels = 2
    models = [
        GCN(in_channels, out_channels),
        GAT(in_channels, out_channels),
        MPNN(in_channels, out_channels),
    ]
    for model in models:
        assert model is not None


def test_training_loop():
    """
    Tests that the training loop runs without errors.
    """
    results = train_and_evaluate(
        model_name="GCN",
        hdf5_path=HDF5_PATH,
        graph_type="reachability",
        epochs=1,
    )
    assert results is not None
    assert "train_loss" in results
    assert "val_loss" in results
