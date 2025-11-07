"""
This module provides the training and evaluation logic for the GNN models
on the SPN dataset. It includes functions for running a training loop,
evaluating model performance, and orchestrating the training process for
different G-NN architectures.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from spn_datasets.gnns.data import SPNDataset
from spn_datasets.gnns.models import GCN, GAT, MPNN


def train_step(model, loader, optimizer):
    """
    Performs a single training step.
    """
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_step(model, loader):
    """
    Performs a single evaluation step.
    """
    model.eval()
    total_loss = 0
    for data in loader:
        out = model(data)
        loss = F.mse_loss(out, data.y)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def train_and_evaluate(
    model_name: str,
    hdf5_path: str,
    graph_type: str,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
):
    """
    Trains and evaluates a specified GNN model on the SPN dataset.

    Args:
        model_name (str): The name of the model to train ('GCN', 'GAT', 'MPNN').
        hdf5_path (str): The path to the HDF5 dataset.
        graph_type (str): The type of graph representation to use ('reachability' or 'petri_net').
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training and evaluation.

    Returns:
        A dictionary containing the training and evaluation results.
    """
    dataset = SPNDataset(hdf5_path=hdf5_path, graph_type=graph_type)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # Determine in_channels and out_channels from the dataset
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes

    # Initialize the model
    if model_name == "GCN":
        model = GCN(in_channels, out_channels)
    elif model_name == "GAT":
        model = GAT(in_channels, out_channels)
    elif model_name == "MPNN":
        model = MPNN(in_channels, out_channels, edge_dim=dataset.num_edge_features)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    results = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_step(model, train_loader, optimizer)
        val_loss = evaluate_step(model, val_loader)
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return results
