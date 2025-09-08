import os
from graphviz import Digraph
import numpy as np
from joblib import Parallel, delayed

np.set_printoptions(precision=4)


def _create_place_nodes(graph, petri_matrix):
    """Creates place nodes for the Petri net graph."""
    num_places = petri_matrix.shape[0]
    for i in range(num_places):
        token_count = petri_matrix[i, -1]
        label = f"P{i+1}\\n" + ("● " * token_count if token_count > 0 else "\\n")
        graph.node(f"P{i+1}", label)


def _create_transition_nodes(graph, num_transitions):
    """Creates transition nodes for the Petri net graph."""
    for i in range(num_transitions):
        graph.node(f"t{i+1}", f"t{i+1}", shape="box")


def _create_edges(graph, petri_matrix, num_transitions):
    """Creates edges between places and transitions."""
    num_places = petri_matrix.shape[0]
    # Input arcs
    for i in range(num_places):
        for j in range(num_transitions):
            if petri_matrix[i, j] == 1:
                graph.edge(f"P{i+1}", f"t{j+1}")
    # Output arcs
    output_matrix = petri_matrix[:, num_transitions:-1]
    for i in range(output_matrix.shape[0]):
        for j in range(output_matrix.shape[1]):
            if output_matrix[i, j] == 1:
                graph.edge(f"t{j+1}", f"P{i+1}")


def plot_petri_net(petri_net_matrix, output_filepath):
    """Generates a visual representation of a Petri net.

    Args:
        petri_net_matrix (np.ndarray): The matrix describing the Petri net.
        output_filepath (str): The path to save the output PNG file.
    """
    graph = Digraph()
    matrix = np.array(petri_net_matrix, dtype=int)
    num_transitions = (matrix.shape[1] - 1) // 2

    _create_place_nodes(graph, matrix)
    _create_transition_nodes(graph, num_transitions)
    _create_edges(graph, matrix, num_transitions)

    graph.format = "png"
    try:
        graph.render(output_filepath, cleanup=True)
    except Exception as e:
        print(f"Error rendering Petri net: {e}")


def plot_reachability_graph(vertices, edges, arc_transitions, output_filepath):
    """Generates a visualization of a reachability graph.

    Args:
        vertices (list): A list of vertices.
        edges (list): A list of edges.
        arc_transitions (list): A list of transition indices for each edge.
        output_filepath (str): The path to save the output PNG file.
    """
    graph = Digraph()
    for i, vertex in enumerate(vertices):
        graph.node(f"M{i}", f"M{i}\\n{vertex}", shape="box")

    for edge, arc_transition in zip(edges, arc_transitions):
        src, dest = f"M{edge[0]}", f"M{edge[1]}"
        label = f"t{arc_transition+1}"
        graph.edge(src, dest, label=label)

    graph.attr(fontsize="20")
    graph.format = "png"
    try:
        graph.render(output_filepath, cleanup=True)
    except Exception as e:
        print(f"Error rendering reachability graph: {e}")


def _format_metrics_label(steady_state_vector, token_density, avg_token_count):
    """Formats the metrics label for the SPN plot."""
    return (
        f"\\nSteady State Probability:\\n{np.array(steady_state_vector)}\\n"
        f"Token Probability Density Function:\\n{np.array(token_density)}\\n"
        f"Average Number of Tokens in Places:\\n{np.array(avg_token_count)}\\n"
        f"Sum of Average Tokens:\\n{np.sum(avg_token_count):.4f}"
    )


def plot_stochastic_petri_net(
    vertices,
    edges,
    arc_transitions,
    lambda_values,
    steady_state_vector,
    token_density,
    avg_token_count,
    output_filepath="test-output/test.gv",
):
    """Generates a visualization of a Stochastic Petri Net (SPN).

    Args:
        vertices (list): A list of vertices.
        edges (list): A list of edges.
        arc_transitions (list): A list of transition indices for each edge.
        lambda_values (list): Firing rates for the transitions.
        steady_state_vector (np.ndarray): Steady-state probability vector.
        token_density (np.ndarray): Token probability density function.
        avg_token_count (np.ndarray): Average number of tokens in each place.
        output_filepath (str, optional): The path to save the output file.
    """
    graph = Digraph()
    for i, vertex in enumerate(vertices):
        graph.node(f"M{i}", f"M{i}\\n{vertex}", shape="box")

    for edge, arc_transition in zip(edges, arc_transitions):
        src, dest = f"M{edge[0]}", f"M{edge[1]}"
        label = f"t{arc_transition+1} [{lambda_values[int(arc_transition)]}]"
        graph.edge(src, dest, label=label)

    metrics_label = _format_metrics_label(steady_state_vector, token_density, avg_token_count)
    graph.attr(label=metrics_label, fontsize="20")
    graph.format = "png"
    try:
        graph.render(output_filepath, cleanup=True)
    except Exception as e:
        print(f"Error rendering SPN: {e}")


def save_visualizations_for_instance(graph_data, output_dir, file_counter):
    """Saves Petri net and SPN visualizations for a single data instance.

    Args:
        graph_data (dict): Data for plotting.
        output_dir (str): Directory to save the images.
        file_counter (int): Counter for unique filenames.
    """
    petri_filepath = os.path.join(output_dir, f"petri_net_{file_counter}")
    spn_filepath = os.path.join(output_dir, f"spn_{file_counter}")

    plot_petri_net(graph_data["petri_net"], petri_filepath)
    plot_stochastic_petri_net(
        graph_data["arr_vlist"],
        graph_data["arr_edge"],
        graph_data["arr_tranidx"],
        graph_data["spn_labda"],
        graph_data["spn_steadypro"],
        graph_data["spn_markdens"],
        graph_data["spn_allmus"],
        spn_filepath,
    )


def visualize_dataset(all_graph_data, output_dir, num_parallel_jobs):
    """Generates and saves visualizations for a collection of data in parallel.

    Args:
        all_graph_data (dict): A dictionary of data instances to visualize.
        output_dir (str): The directory to save the images.
        num_parallel_jobs (int): The number of parallel jobs to run.
    """
    Parallel(n_jobs=num_parallel_jobs)(
        delayed(save_visualizations_for_instance)(graph_data, output_dir, i + 1)
        for i, graph_data in enumerate(all_graph_data.values())
    )
    print("Image saving process initiated successfully!")
