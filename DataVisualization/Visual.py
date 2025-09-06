from graphviz import Digraph
import numpy as np
import os
from joblib import Parallel, delayed

np.set_printoptions(precision=4)


def plot_petri(petri_net_matrix, output_filepath):
    """Generates a visual representation of a Petri net from a matrix and saves it as a PNG image.

    The input matrix defines the structure of the Petri net. It is expected to be structured as follows:
    - The first part of the matrix represents the connections from places to transitions.
    - The middle part represents the connections from transitions to places.
    - The last column represents the initial marking (tokens) of the places.

    Args:
        petri_net_matrix (list or np.ndarray): A matrix describing the Petri net structure and initial marking.
        output_filepath (str): The path (including filename without extension) where the output PNG file will be saved.
    """
    dot = Digraph()
    matrix = np.array(petri_net_matrix, dtype=int)

    # The divider index splits the matrix into input and output connections for transitions.
    # It assumes the last column is for markings, so it's subtracted.
    divider_index = int((matrix.shape[1] - 1) / 2)

    # Create place nodes (circles)
    num_places = len(matrix)
    for i in range(num_places):
        # The last column in the matrix represents the number of tokens in the place.
        token_count = matrix[i][-1]
        node_label = f"P{i + 1}\\n"
        if token_count >= 1:
            # Represent tokens with '●' symbols
            node_label += "● " * token_count
        else:
            # Add space for alignment if there are no tokens
            node_label += "\\n"

        dot.node(f"P{i + 1}", node_label)

    # Create transition nodes (boxes)
    num_transitions = divider_index
    for i in range(num_transitions):
        dot.node(f"t{i + 1}", f"t{i + 1}", shape="box")

    # Create edges from places to transitions (input arcs)
    for i in range(num_places):
        for j in range(num_transitions):
            if matrix[i][j] == 1:
                dot.edge(f"P{i + 1}", f"t{j + 1}")

    # Extract the part of the matrix representing connections from transitions to places.
    output_matrix = matrix[:, divider_index:-1]

    # Create edges from transitions to places (output arcs)
    for i in range(len(output_matrix)):
        for j in range(output_matrix.shape[1]):
            if output_matrix[i][j] == 1:
                dot.edge(f"t{j + 1}", f"P{i + 1}")

    dot.format = "png"
    try:
        # cleanup=True removes the intermediate DOT source file after rendering.
        dot.render(output_filepath, cleanup=True)
    except Exception:
        # Fail silently if graphviz is not installed or there's another issue.
        return


def plot_arri_gra(vertex_list, edge_list, arc_transition_list, output_filepath):
    """Generates a visualization of a graph, likely an arrival graph, and saves it as a PNG image.

    This function creates a graph with nodes representing states (Markings) and edges representing transitions between them.

    Args:
        vertex_list (list): A list of vertices, where each vertex has a label or value.
        edge_list (list of tuples): A list of edges, where each edge is a tuple of two vertex indices (0-based).
        arc_transition_list (list): A list of transition indices corresponding to each edge.
        output_filepath (str): The path (including filename without extension) where the output PNG file will be saved.
    """
    dot = Digraph()

    # Create nodes for each vertex in the list
    for i, vertex_value in enumerate(vertex_list):
        node_label = f"M{i}\\n{vertex_value}"
        dot.node(f"M{i + 1}", node_label, shape="box")

    # Create edges between nodes
    for edge, arc_transition in zip(edge_list, arc_transition_list):
        source_node = f"M{edge[0] + 1}"
        destination_node = f"M{edge[1] + 1}"
        edge_label = f"t{arc_transition + 1}"
        dot.edge(source_node, destination_node, label=edge_label)

    dot.attr(fontsize="20")
    dot.format = "png"
    try:
        # cleanup=True removes the intermediate DOT source file after rendering.
        dot.render(output_filepath, cleanup=True)
    except Exception:
        # Fail silently if there's an issue with rendering.
        return


def plot_spn(
    vertex_list,
    edge_list,
    arc_transition_list,
    lambda_values,
    steady_state_vector,
    token_density_function,
    average_token_count,
    output_filepath="test-output/test.gv",
):
    """
    Generates a visualization of a Stochastic Petri Net (SPN) and related performance metrics.

    This function creates a graph of the SPN's state space and annotates it with lambda values.
    It also adds a label to the graph with key metrics like steady-state probability,
    token density, and average token counts.

    Args:
        vertex_list (list): A list of vertices, where each vertex has a label or value.
        edge_list (list of tuples): A list of edges, where each edge is a tuple of two vertex indices.
        arc_transition_list (list): A list of transition indices corresponding to each edge.
        lambda_values (list): A list of lambda values (firing rates) for the transitions.
        steady_state_vector (list or np.ndarray): The steady-state probability vector for the markings.
        token_density_function (list or np.ndarray): The token probability density function.
        average_token_count (list or np.ndarray): The average number of tokens in each place.
        output_filepath (str, optional): The path where the output PNG file will be saved.
                                        Defaults to "test-output/test.gv".
    """
    dot = Digraph()

    # Create nodes for each vertex
    for i, vertex_value in enumerate(vertex_list):
        node_label = f"M{i}\\n{vertex_value}"
        dot.node(f"M{i + 1}", node_label, shape="box")

    # Create edges with transition labels and lambda values
    for edge, arc_transition in zip(edge_list, arc_transition_list):
        transition_index = int(arc_transition)
        source_node = f"M{edge[0] + 1}"
        destination_node = f"M{edge[1] + 1}"
        edge_label = f"t{arc_transition + 1} [{lambda_values[transition_index]}]"
        dot.edge(source_node, destination_node, label=edge_label)

    # Create a detailed label with performance metrics
    metrics_label = (
        f"\\n Steady State Probability: \\n{np.array(steady_state_vector)}\\n"
        f" Token Probability Density Function:\\n{np.array(token_density_function)}\\n"
        f" The Average Number of Tokens in the Place :\\n{np.array(average_token_count)}\\n"
        f" Sum of the Average Numbers of Tokens:\\n{np.array([np.sum(average_token_count)])}"
    )
    dot.attr(label=metrics_label)
    dot.attr(fontsize="20")
    dot.format = "png"

    try:
        dot.render(output_filepath, cleanup=True)
    except Exception:
        return


def save_i_pic(graph_data, output_directory, file_counter):
    """
    Generates and saves Petri net and SPN visualizations for a single data instance.

    This function calls the respective plotting functions for the Petri net and the SPN
    state space, which save the visualizations to the specified directory.

    Args:
        graph_data (dict): A dictionary containing the data required for plotting.
                           Expected keys include 'petri_net', 'arr_vlist', 'arr_edge', etc.
        output_directory (str): The directory where the final PNG images will be saved.
        file_counter (int): A counter to create unique filenames for the images.
    """
    # Define filepaths for the output images. The plotting functions will add the .png extension.
    petri_filepath = os.path.join(output_directory, f"data(petri){file_counter}")
    spn_filepath = os.path.join(output_directory, f"data(arr){file_counter}")

    # Generate and save the Petri net visualization
    plot_petri(
        petri_net_matrix=graph_data["petri_net"],
        output_filepath=petri_filepath
    )

    # Generate and save the SPN visualization
    plot_spn(
        vertex_list=graph_data["arr_vlist"],
        edge_list=graph_data["arr_edge"],
        arc_transition_list=graph_data["arr_tranidx"],
        lambda_values=graph_data["spn_labda"],
        steady_state_vector=graph_data["spn_steadypro"],
        token_density_function=graph_data["spn_markdens"],
        average_token_count=graph_data["spn_allmus"],
        output_filepath=spn_filepath,
    )


def visual_data(all_graph_data, output_directory, parallel_job_count):
    """
    Generates and saves visualizations for a collection of data in parallel.

    This function uses joblib to parallelize the process of creating and saving
    Petri net and SPN visualizations for each data item in the input collection.

    Args:
        all_graph_data (dict): A dictionary containing all the data instances to be visualized.
                               It is expected to have keys like 'data1', 'data2', etc.
        output_directory (str): The directory where the final PNG images will be saved.
        parallel_job_count (int): The number of parallel jobs to run for generating images.
    """
    Parallel(n_jobs=parallel_job_count)(
        delayed(save_i_pic)(
            graph_data=all_graph_data[f"data{i + 1}"],
            output_directory=output_directory,
            file_counter=i + 1,
        )
        for i in range(len(all_graph_data))
    )
    print("Image saving process initiated successfully!")
