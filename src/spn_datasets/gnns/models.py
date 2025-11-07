"""
This module defines the GNN models for the SPN-Benchmarks project, including
GCN, GAT, and a simple Message-Passing GNN. These models are designed to
predict SPN properties from graph representations of Petri nets or their
reachability graphs.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, MessagePassing, global_mean_pool


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) for graph-level prediction.

    This model uses two GCN layers to learn node embeddings, followed by a
    global mean pooling layer to obtain a graph embedding, and a final

    linear layer for prediction.
    """

    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.bn1 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 128)
        self.bn2 = BatchNorm1d(128)
        self.out = Linear(128, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))

        # Global mean pooling to get a graph-level embedding
        x = global_mean_pool(x, batch)

        return self.out(x)


class GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) for graph-level prediction.

    This model uses two GAT layers with multi-head attention to learn
    node embeddings, followed by a global mean pooling and a linear layer.
    """

    def __init__(self, in_channels, out_channels, heads=4):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, 32, heads=heads)
        self.bn1 = BatchNorm1d(32 * heads)
        self.gat2 = GATConv(32 * heads, 32, heads=heads)
        self.bn2 = BatchNorm1d(32 * heads)
        self.out = Linear(32 * heads, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.gat1(x, edge_index)))
        x = F.relu(self.bn2(self.gat2(x, edge_index)))

        x = global_mean_pool(x, batch)
        return self.out(x)


class MPNN(MessagePassing):
    """
    A simple Message-Passing Graph Neural Network.

    This model implements a basic message-passing scheme where messages
    are computed by a neural network and aggregated by summing.
    """

    def __init__(self, in_channels, out_channels, edge_dim=1):
        super(MPNN, self).__init__(aggr="mean")
        self.edge_net = Sequential(Linear(in_channels * 2 + edge_dim, 128), ReLU())
        self.node_net = Sequential(Linear(in_channels + 128, 128), ReLU())
        self.out = Linear(128, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Propagate messages and update node embeddings
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.node_net(torch.cat([data.x, x], dim=1))

        # Global pooling and final prediction
        x = global_mean_pool(x, batch)
        return self.out(x)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features, x_j: source node features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_net(msg_input)
