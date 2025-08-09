#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : Visual.py
@Date    : 2020-09-21
@Author  : mingjian

This module provides functions to visualize Stochastic Petri Nets (SPNs) and their
corresponding reachability graphs using the Graphviz library.
"""

import os
from typing import List, Dict, Any

import numpy as np
from graphviz import Digraph
from joblib import Parallel, delayed

np.set_printoptions(precision=4)


def plot_petri_net(petri_matrix: np.ndarray, output_filepath: str):
    """
    Generates a visual representation of a Petri net structure.

    Args:
        petri_matrix: The matrix representation of the Petri net.
        output_filepath: The path (without extension) to save the output image.
    """
    dot = Digraph("PetriNet")
    petri_matrix = np.array(petri_matrix, dtype=int)
    num_places, num_cols = petri_matrix.shape
    num_transitions = (num_cols - 1) // 2

    # Create nodes for places, showing tokens for the initial marking
    for i in range(num_places):
        initial_tokens = petri_matrix[i, -1]
        place_label = f"P{i+1}"
        if initial_tokens > 0:
            token_str = "\\n" + "● " * initial_tokens
            place_label += token_str
        dot.node(f"P{i+1}", place_label, shape="circle")

    # Create nodes for transitions
    for i in range(num_transitions):
        dot.node(f"t{i+1}", f"t{i+1}", shape="box")

    # Create edges based on the pre and post matrices
    pre_matrix = petri_matrix[:, :num_transitions]
    post_matrix = petri_matrix[:, num_transitions:-1]

    for i in range(num_places):
        for j in range(num_transitions):
            # Arcs from places to transitions
            if pre_matrix[i, j] == 1:
                dot.edge(f"P{i+1}", f"t{j+1}")
            # Arcs from transitions to places
            if post_matrix[i, j] == 1:
                dot.edge(f"t{j+1}", f"P{i+1}")

    dot.format = "png"
    try:
        dot.render(output_filepath, cleanup=True)
    except Exception as e:
        print(f"Error rendering Petri net graph: {e}")


def plot_reachability_graph(
    markings: List[np.ndarray],
    edges: List[List[int]],
    arc_transitions: List[int],
    output_filepath: str,
):
    """
    Generates a visual representation of a reachability graph.

    Args:
        markings: A list of reachable markings (states).
        edges: A list of edges [from_idx, to_idx] between markings.
        arc_transitions: The transition index corresponding to each edge.
        output_filepath: The path (without extension) to save the output image.
    """
    dot = Digraph("ReachabilityGraph")
    for i, marking in enumerate(markings):
        dot.node(f"M{i}", f"M{i}\\n{marking}", shape="box")

    for edge, transition_idx in zip(edges, arc_transitions):
        dot.edge(
            f"M{edge[0]}", f"M{edge[1]}", label=f"t{transition_idx + 1}"
        )
    dot.attr(fontsize="20")
    dot.format = "png"
    try:
        dot.render(output_filepath, cleanup=True)
    except Exception as e:
        print(f"Error rendering reachability graph: {e}")


def plot_stochastic_petri_net(
    markings: List[np.ndarray],
    edges: List[List[int]],
    arc_transitions: List[int],
    firing_rates: np.ndarray,
    steady_state_probs: np.ndarray,
    marking_densities: np.ndarray,
    avg_markings: np.ndarray,
    output_filepath: str,
):
    """
    Visualizes a solved SPN, including its properties.

    Args:
        markings: List of reachable markings.
        edges: List of edges in the reachability graph.
        arc_transitions: Transition for each edge.
        firing_rates: Firing rate for each transition.
        steady_state_probs: Steady-state probability for each marking.
        marking_densities: Token probability density function.
        avg_markings: Average number of tokens in each place.
        output_filepath: Path to save the output image.
    """
    dot = Digraph("SPN")
    for i, marking in enumerate(markings):
        dot.node(f"M{i}", f"M{i}\\n{marking}", shape="box")

    for edge, trans_idx in zip(edges, arc_transitions):
        rate = firing_rates[trans_idx]
        dot.edge(f"M{edge[0]}", f"M{edge[1]}", label=f"t{trans_idx+1} [{rate}]")

    # Create a formatted label with SPN properties
    info_label = (
        f"\\nSteady State Probability:\\n{np.array2string(steady_state_probs)}\\n\\n"
        f"Token Probability Density Function:\\n{np.array2string(marking_densities)}\\n\\n"
        f"Average Number of Tokens in Places:\\n{np.array2string(avg_markings)}\\n\\n"
        f"Sum of Average Tokens:\\n{np.sum(avg_markings):.4f}"
    )
    dot.attr(label=info_label, fontsize="20")
    dot.format = "png"
    try:
        dot.render(output_filepath, cleanup=True)
    except Exception as e:
        print(f"Error rendering SPN graph: {e}")


def save_spn_visualizations(
    spn_data: Dict[str, Any], output_directory: str, file_counter: int
):
    """
    Saves visualizations for a single SPN data sample.

    Args:
        spn_data: A dictionary containing all data for a single SPN.
        output_directory: The directory to save the images in.
        file_counter: An integer to create unique filenames.
    """
    petri_filepath = os.path.join(output_directory, f"petri_net_{file_counter}")
    spn_filepath = os.path.join(output_directory, f"spn_graph_{file_counter}")

    plot_petri_net(spn_data["petri_net"], petri_filepath)
    plot_stochastic_petri_net(
        spn_data["arr_vlist"],
        spn_data["arr_edge"],
        spn_data["arr_tranidx"],
        spn_data["spn_labda"],
        spn_data["spn_steadypro"],
        spn_data["spn_markdens"],
        spn_data["spn_allmus"],
        spn_filepath,
    )


def visualize_dataset_in_parallel(
    dataset: Dict[str, Any], output_directory: str, num_parallel_jobs: int
):
    """
    Generates and saves visualizations for an entire dataset in parallel.

    Args:
        dataset: A dictionary where keys are data sample names (e.g., "data1").
        output_directory: The directory where images will be saved.
        num_parallel_jobs: The number of parallel jobs to use for rendering.
    """
    print(f"Starting visualization of {len(dataset)} samples...")
    Parallel(n_jobs=num_parallel_jobs)(
        delayed(save_spn_visualizations)(
            dataset[f"data{i+1}"], output_directory, i + 1
        )
        for i in range(len(dataset))
    )
    print("Visualization generation successful!")
