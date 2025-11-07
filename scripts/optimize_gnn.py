"""
This script optimizes GNN hyperparameters using Optuna.
"""

import toml
import argparse
import pytorch_lightning as pl
import optuna
from functools import partial

from spn_datasets.gnns.lightning_data import SPNDataModule
from spn_datasets.gnns.lightning_models import GCNLightning, GATLightning, MPNNLightning


def objective(trial, config):
    """
    Objective function for Optuna to optimize.
    """
    # Suggest hyperparameters
    model_name = trial.suggest_categorical("model_name", ["GCN", "GAT", "MPNN"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # Initialize the DataModule
    data_module = SPNDataModule(
        hdf5_path=config["dataset"]["hdf5_path"],
        graph_type=config["dataset"]["graph_type"],
        batch_size=batch_size,
        train_val_test_split=tuple(config["dataset"]["train_val_test_split"]),
    )

    # Call setup to initialize the dataset and get dimensions
    data_module.setup()
    in_channels = data_module.dataset.num_node_features
    out_channels = data_module.dataset.num_classes
    edge_dim = data_module.dataset.num_edge_features

    # Initialize the model
    if model_name == "GCN":
        model = GCNLightning(in_channels, out_channels, learning_rate=learning_rate)
    elif model_name == "GAT":
        heads = trial.suggest_int("heads", 1, 8)
        model = GATLightning(in_channels, out_channels, heads=heads, learning_rate=learning_rate)
    elif model_name == "MPNN":
        model = MPNNLightning(in_channels, out_channels, edge_dim=edge_dim, learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    # Train the model
    trainer.fit(model, data_module)

    return trainer.callback_metrics["val_loss"].item()


def main():
    parser = argparse.ArgumentParser(description="Optimize GNN hyperparameters using Optuna.")
    parser.add_argument("--config", type=str, required=True, help="Path to the TOML configuration file.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials.")
    parser.add_argument("--output-file", type=str, default="best_params.toml", help="Path to save the best hyperparameters.")
    args = parser.parse_args()

    config = toml.load(args.config)

    study = optuna.create_study(direction="minimize")
    objective_func = partial(objective, config=config)
    study.optimize(objective_func, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    with open(args.output_file, "w") as f:
        toml.dump({"training": trial.params}, f)
    print(f"Best hyperparameters saved to {args.output_file}")


if __name__ == "__main__":
    main()
