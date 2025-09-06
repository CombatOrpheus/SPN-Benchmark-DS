"""
This script trains a Graph Neural Network (GNN) for graph regression tasks
on randomly generated SPN data.
"""

import argparse
import json
import os
import time
import random
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from GNNs.nets.load_net import gnn_model
from GNNs.train.train_graph_regression import train_epoch, evaluate_network
from GNNs.datasets.NetLearningDatasetDGL import NetLearningDatasetDGL


def setup_gpu(use_gpu, gpu_id):
    """Sets up the GPU for training."""
    if torch.cuda.is_available() and use_gpu:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_model_params(model_name, net_params):
    """Calculates the total number of parameters in a model."""
    model = gnn_model(model_name, net_params)
    return sum(p.numel() for p in model.parameters())


def get_output_dirs(out_dir, model_name, dataset_name, gpu_id):
    """Creates and returns the output directories for logs, checkpoints, and results."""
    timestamp = time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    base_dir = f"{model_name}_{dataset_name}_GPU{gpu_id}_{timestamp}"

    log_dir = os.path.join(out_dir, "logs", base_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints", base_dir)
    result_dir = os.path.join(out_dir, "results")
    config_dir = os.path.join(out_dir, "configs")

    for d in [log_dir, ckpt_dir, result_dir, config_dir]:
        os.makedirs(d, exist_ok=True)

    result_file = os.path.join(result_dir, f"result_{base_dir}.txt")
    config_file = os.path.join(config_dir, f"config_{base_dir}.txt")

    return log_dir, ckpt_dir, result_file, config_file


def write_config_to_file(config_file, dataset_name, model_name, params, net_params):
    """Writes the configuration to a text file."""
    with open(config_file, "w") as f:
        f.write(
            f"Dataset: {dataset_name},\nModel: {model_name}\n\n"
            f"params={params}\n\nnet_params={net_params}\n\n"
            f"Total Parameters: {net_params['total_param']}\n\n"
        )


def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, device, params, writer, ckpt_dir):
    """The main training and evaluation loop."""
    start_time = time.time()
    per_epoch_time = []

    for epoch in tqdm(range(params["epochs"]), desc="Epochs"):
        epoch_start_time = time.time()

        train_loss, train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
        writer.add_scalar("train/_loss", train_loss, epoch)
        writer.add_scalar("train/_mae", train_mae, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        _, test_mae, _ = evaluate_network(model, device, test_loader, epoch)

        tqdm.write(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

        per_epoch_time.append(time.time() - epoch_start_time)

        # Checkpoint saving
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pkl"))

        # Prune old checkpoints
        old_checkpoints = glob.glob(os.path.join(ckpt_dir, "epoch_*.pkl"))
        for ckpt in old_checkpoints:
            epoch_num = int(ckpt.split("_")[-1].split(".")[0])
            if epoch_num < epoch - 1:
                os.remove(ckpt)

        if optimizer.param_groups[0]["lr"] < params["min_lr"]:
            print("Learning rate has reached the minimum value. Stopping training.")
            break

        if time.time() - start_time > params["max_time"] * 3600:
            print(f"Maximum training time of {params['max_time']} hours reached. Stopping training.")
            break

    final_train_mae, final_train_err = evaluate_network(model, device, train_loader, epoch)
    final_test_mae, final_test_err = evaluate_network(model, device, test_loader, epoch)

    return final_train_mae, final_train_err, final_test_mae, final_test_err, time.time() - start_time, np.mean(per_epoch_time)


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description="GNN Training for Graph Regression on Random SPN Data")
    parser.add_argument("--config", default="config/GNNConfig/RandData/SPN_CNN_Regression.json", help="Path to config file.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = setup_gpu(config["gpu"]["use"], config["gpu"]["id"])
    config["net_params"]["device"] = device

    model_name = config["model"]
    dataset_name = config["dataset"]
    data_dir = config["data_dir"] % dataset_name
    out_dir = config["out_dir"] % dataset_name
    params = config["params"]
    net_params = config["net_params"]
    net_params["gpu_id"] = config["gpu"]["id"]
    net_params["batch_size"] = params["batch_size"]

    dataset = NetLearningDatasetDGL(data_dir)
    dataset.name = dataset_name

    log_dir, ckpt_dir, result_file, config_file = get_output_dirs(out_dir, model_name, dataset_name, config["gpu"]["id"])

    net_params["total_param"] = get_model_params(model_name, net_params)
    write_config_to_file(config_file, dataset_name, model_name, params, net_params)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "RUN_0"))

    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(params["seed"])

    print(f"Training Graphs: {len(dataset.train)}, Test Graphs: {len(dataset.test)}")

    model = gnn_model(model_name, net_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["init_lr"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=params["lr_reduce_factor"], patience=params["lr_schedule_patience"], verbose=True
    )

    train_loader = DataLoader(dataset.train, batch_size=params["batch_size"], shuffle=True, collate_fn=dataset.collate)
    test_loader = DataLoader(dataset.test, batch_size=params["batch_size"], shuffle=False, collate_fn=dataset.collate)

    try:
        train_mae, train_err, test_mae, test_err, total_time, avg_epoch_time = train_and_evaluate(
            model, train_loader, test_loader, optimizer, scheduler, device, params, writer, ckpt_dir
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return

    print(f"Test MAE: {test_mae:.4f}, Train MAE: {train_mae:.4f}")
    print(f"Test Error Rate: {test_err:.4f}, Train Error Rate: {train_err:.4f}")
    print(f"Total Time: {total_time/3600:.2f} hours, Avg Time/Epoch: {avg_epoch_time:.2f}s")

    with open(result_file, "w") as f:
        f.write(
            f"Dataset: {dataset_name}\nModel: {model_name}\n\n"
            f"FINAL RESULTS\n"
            f"Test MAE: {test_mae:.4f}\nTrain MAE: {train_mae:.4f}\n"
            f"Test Error Rate: {test_err:.4f}\nTrain Error Rate: {train_err:.4f}\n\n"
            f"Total Time: {total_time/3600:.2f} hours\nAvg Time/Epoch: {avg_epoch_time:.2f}s"
        )

    writer.close()


if __name__ == "__main__":
    main()
