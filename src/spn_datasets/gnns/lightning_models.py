"""
This module provides PyTorch Lightning versions of the GNN models
for the SPN-Benchmarks project.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from spn_datasets.gnns.models import GCN, GAT, MPNN


class GNNLightningModule(pl.LightningModule):
    """
    A generic PyTorch Lightning module for GNN models.
    """

    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.l1_loss(out, batch.y)  # MAE Loss
        mae = self.mae(out, batch.y)
        mape = self.mape(out, batch.y)
        self.log('train_loss', loss)
        self.log('train_mae', mae)
        self.log('train_mape', mape)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.l1_loss(out, batch.y)  # MAE Loss
        mae = self.mae(out, batch.y)
        mape = self.mape(out, batch.y)
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        self.log('val_mape', mape)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class GCNLightning(GNNLightningModule):
    def __init__(self, in_channels, out_channels, learning_rate=0.001):
        model = GCN(in_channels, out_channels)
        super().__init__(model, learning_rate)


class GATLightning(GNNLightningModule):
    def __init__(self, in_channels, out_channels, heads=4, learning_rate=0.001):
        model = GAT(in_channels, out_channels, heads=heads)
        super().__init__(model, learning_rate)


class MPNNLightning(GNNLightningModule):
    def __init__(self, in_channels, out_channels, edge_dim=1, learning_rate=0.001):
        model = MPNN(in_channels, out_channels, edge_dim=edge_dim)
        super().__init__(model, learning_rate)
