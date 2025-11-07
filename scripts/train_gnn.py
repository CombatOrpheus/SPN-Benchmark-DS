"""
This script trains a GNN model using PyTorch Lightning.
"""

import toml
import argparse
import pytorch_lightning as pl
from spn_datasets.gnns.lightning_data import SPNDataModule
from spn_datasets.gnns.lightning_models import GCNLightning, GATLightning, MPNNLightning


def main():
    parser = argparse.ArgumentParser(description="Train a GNN model using PyTorch Lightning.")
    parser.add_argument("--config", type=str, required=True, help="Path to the TOML configuration file.")
    args = parser.parse_args()

    config = toml.load(args.config)

    # Initialize the DataModule
    data_module = SPNDataModule(
        hdf5_path=config["dataset"]["hdf5_path"],
        graph_type=config["dataset"]["graph_type"],
        batch_size=config["training"]["batch_size"],
        train_val_test_split=tuple(config["dataset"]["train_val_test_split"]),
    )

    # Call setup to initialize the dataset and get dimensions
    data_module.setup()
    in_channels = data_module.dataset.num_node_features
    out_channels = data_module.dataset.num_classes
    edge_dim = data_module.dataset.num_edge_features

    # Initialize the model
    if config["training"]["model_name"] == "GCN":
        model = GCNLightning(in_channels, out_channels, learning_rate=config["training"]["learning_rate"])
    elif config["training"]["model_name"] == "GAT":
        model = GATLightning(in_channels, out_channels, learning_rate=config["training"]["learning_rate"])
    elif config["training"]["model_name"] == "MPNN":
        model = MPNNLightning(in_channels, out_channels, edge_dim=edge_dim, learning_rate=config["training"]["learning_rate"])
    else:
        raise ValueError(f"Unknown model: {config['training']['model_name']}")

    # Initialize the Trainer
    trainer = pl.Trainer(max_epochs=config["training"]["epochs"])

    # Train the model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
