"""
This module provides the SPNDataset class for loading SPN data and converting
it into a format suitable for GNNs using PyTorch Geometric. It supports
creating graph representations from either the reachability graph or the
underlying Petri net.
"""

import torch
from torch_geometric.data import Data, Dataset
import numpy as np

from spn_datasets.utils.HDF5Reader import SPNDataReader


def create_reachability_graph_data(sample: dict) -> Data:
    """
    Creates a PyTorch Geometric Data object from the reachability graph of an SPN.

    Args:
        sample (dict): A dictionary containing the data for a single SPN sample.

    Returns:
        A torch_geometric.data.Data object representing the reachability graph.
    """
    # Use a constant feature for each node, as the markings have variable length
    num_nodes = len(sample["arr_vlist"])
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    # Edge index represents the transitions between states
    edge_index = torch.tensor(sample["arr_edge"].T, dtype=torch.long)

    # Edge attributes are the firing rates of the transitions causing the state change
    firing_rates = sample["spn_labda"]
    transition_indices = sample["arr_tranidx"]
    edge_attr = torch.tensor(firing_rates[transition_indices], dtype=torch.float).view(-1, 1)

    # Graph-level targets: total average tokens (spn_mu) and mean firing rate
    spn_mu = torch.tensor([sample["spn_mu"]], dtype=torch.float)
    mean_firing_rate = torch.tensor([np.mean(sample["spn_labda"])], dtype=torch.float)
    y = torch.cat([spn_mu, mean_firing_rate]).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class SPNDataset(Dataset):
    """
    A PyTorch Geometric dataset for SPN data.

    This dataset can load data from an HDF5 file and convert each sample
    into a graph representation suitable for GNNs. It supports two modes:
    'reachability' for using the reachability graph and 'petri_net' for
    the bipartite Petri net graph.

    Args:
        hdf5_path (str): The path to the HDF5 file.
        graph_type (str): The type of graph to create ('reachability' or 'petri_net').
    """

    def __init__(self, hdf5_path: str, graph_type: str = "reachability"):
        super().__init__()
        if graph_type not in ["reachability", "petri_net"]:
            raise ValueError("graph_type must be 'reachability' or 'petri_net'")

        self.hdf5_path = hdf5_path
        self.graph_type = graph_type
        self.reader = SPNDataReader(self.hdf5_path)

        with self.reader as r:
            self._len = len(r)

        # Determine dataset properties from the first sample
        if self._len > 0:
            sample_data = self.get(0)
            self._num_node_features = sample_data.num_node_features
            self._num_classes = sample_data.y.shape[1] if sample_data.y is not None else 0
            self._num_edge_features = sample_data.num_edge_features
        else:
            self._num_node_features = 0
            self._num_classes = 0
            self._num_edge_features = 0

    @property
    def num_node_features(self) -> int:
        return self._num_node_features

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_edge_features(self) -> int:
        return self._num_edge_features

    def len(self) -> int:
        return self._len

    def get(self, idx: int) -> Data:
        """
        Retrieves a single graph sample from the dataset.
        """
        with self.reader as r:
            sample = r.get_sample(idx)

        if self.graph_type == "reachability":
            return create_reachability_graph_data(sample)
        else:
            return create_petri_net_graph_data(sample)


def create_petri_net_graph_data(sample: dict) -> Data:
    """
    Creates a PyTorch Geometric Data object from the bipartite graph of a Petri net.

    Args:
        sample (dict): A dictionary containing the data for a single SPN sample.

    Returns:
        A torch_geometric.data.Data object representing the Petri net.
    """
    petri_net = sample["petri_net"]
    num_places = petri_net.shape[0]
    num_transitions = (petri_net.shape[1] - 1) // 2

    pre_matrix = petri_net[:, :num_transitions]
    post_matrix = petri_net[:, num_transitions : 2 * num_transitions]
    initial_marking = petri_net[:, -1]
    firing_rates = sample["spn_labda"]

    # Node features: Places are one type, transitions are another
    place_features = torch.zeros((num_places, 2))
    place_features[:, 0] = 1  # Type 0 for places
    initial_marking_tensor = torch.tensor(initial_marking, dtype=torch.float).view(-1, 1)

    transition_features = torch.zeros((num_transitions, 2))
    transition_features[:, 1] = 1  # Type 1 for transitions
    firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float).view(-1, 1)

    # Combine features with their primary attributes (initial markings and firing rates)
    place_features_combined = torch.cat([place_features, initial_marking_tensor], dim=1)
    transition_features_combined = torch.cat([transition_features, firing_rates_tensor], dim=1)
    x = torch.cat([place_features_combined, transition_features_combined], dim=0)

    # Edges for the bipartite graph
    src_nodes, dst_nodes = [], []
    edge_weights = []

    # Edges from places to transitions
    place_indices, transition_indices_pre = np.where(pre_matrix > 0)
    src_nodes.extend(place_indices)
    dst_nodes.extend(transition_indices_pre + num_places)
    edge_weights.extend(pre_matrix[place_indices, transition_indices_pre])

    # Edges from transitions to places
    place_indices_post, transition_indices_post = np.where(post_matrix > 0)
    src_nodes.extend(transition_indices_post + num_places)
    dst_nodes.extend(place_indices_post)
    edge_weights.extend(post_matrix[place_indices_post, transition_indices_post])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)

    # Graph-level targets: total average tokens (spn_mu) and mean firing rate
    spn_mu = torch.tensor([sample["spn_mu"]], dtype=torch.float)
    mean_firing_rate = torch.tensor([np.mean(sample["spn_labda"])], dtype=torch.float)
    y = torch.cat([spn_mu, mean_firing_rate]).unsqueeze(0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
