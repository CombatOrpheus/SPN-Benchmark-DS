import pytest
from spn_datasets.generator.PetriGenerate import (
    generate_random_petri_net,
    prune_petri_net,
    add_missing_connections,
    delete_excess_edges,
)
from spn_datasets.generator.ArrivableGraph import generate_reachability_graph, get_enabled_transitions
from spn_datasets.generator.SPN import solve_for_steady_state, is_connected, compute_qualitative_properties
from spn_datasets.generator.DataTransformation import generate_petri_net_variations
import numpy as np
from scipy.sparse import csc_array
from memory_profiler import memory_usage


def test_benchmark_generate_random_petri_net(benchmark):
    """Benchmark the generation of a random Petri net."""

    def f():
        generate_random_petri_net(10, 5)

    benchmark(f)


def test_benchmark_generate_reachability_graph(benchmark):
    """Benchmark the generation of a reachability graph."""
    petri_net = generate_random_petri_net(10, 5)

    def f():
        generate_reachability_graph(petri_net)

    benchmark(f)


from scipy.sparse import csr_array

def test_benchmark_solve_for_steady_state(benchmark):
    """Benchmark solving for steady-state probabilities."""
    A_sq = csr_array(np.random.rand(100, 100))
    b_sq = np.zeros(100)
    b_sq[-1] = 1.0

    def f():
        solve_for_steady_state(A_sq, b_sq)

    benchmark(f)


def test_benchmark_generate_petri_net_variations(benchmark):
    """Benchmark the generation of Petri net variations."""
    petri_matrix = generate_random_petri_net(10, 5)
    config = {
        "enable_add_edge": True,
        "enable_delete_edge": True,
        "enable_add_place": True,
        "enable_add_token": True,
        "enable_delete_token": True,
        "enable_rate_variations": True,
        "number_of_parallel_jobs": -1,
    }

    def f():
        generate_petri_net_variations(petri_matrix, config)

    benchmark(f)


def test_memory_usage_generate_reachability_graph():
    """Test the memory usage of generate_reachability_graph."""
    petri_net = generate_random_petri_net(10, 5)
    mem_usage = memory_usage((generate_reachability_graph, (petri_net,)), max_usage=True)
    assert mem_usage < 400  # Set a baseline memory limit in MB


def test_benchmark_prune_petri_net(benchmark):
    # Setup a matrix
    petri_matrix = generate_random_petri_net(50, 50)

    def f():
        m = petri_matrix.copy()
        prune_petri_net(m)

    benchmark(f)


def test_benchmark_get_enabled_transitions(benchmark):
    num_places = 50
    num_transitions = 50
    pre_condition_matrix = np.random.randint(0, 3, size=(num_places, num_transitions)).astype(np.int64)
    change_matrix = np.random.randint(-1, 2, size=(num_places, num_transitions)).astype(np.int64)
    current_marking_vector = np.random.randint(0, 5, size=(num_places,)).astype(np.int64)

    # warmup
    get_enabled_transitions(pre_condition_matrix, change_matrix, current_marking_vector)

    def f():
        get_enabled_transitions(pre_condition_matrix, change_matrix, current_marking_vector)

    benchmark(f)


def test_benchmark_add_missing_connections(benchmark):
    petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)
    # Zero out some to trigger the missing connections
    petri_matrix[:, 5:10] = 0
    petri_matrix[20:25, :] = 0

    def f():
        m = petri_matrix.copy()
        add_missing_connections(m, 50)

    benchmark(f)


def test_benchmark_delete_excess_edges(benchmark):
    petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)

    def f():
        m = petri_matrix.copy()
        delete_excess_edges(m, 50)

    benchmark(f)


def test_benchmark_is_connected(benchmark):
    petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)

    def f():
        is_connected(petri_matrix)

    benchmark(f)


def test_benchmark_compute_qualitative_properties(benchmark):
    vertices = np.array([np.random.randint(0, 5, size=(50,)).astype(np.int32) for _ in range(500)])
    edges = np.array([[i, (i + 1) % 500] for i in range(500)])

    def f():
        compute_qualitative_properties(vertices, edges)

    benchmark(f)
