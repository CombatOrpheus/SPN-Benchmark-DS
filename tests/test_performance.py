import pytest
from DataGenerate.PetriGenerate import generate_random_petri_net
from DataGenerate.ArrivableGraph import generate_reachability_graph
from DataGenerate.SPN import solve_for_steady_state
from DataGenerate.DataTransformation import generate_petri_net_variations
import numpy as np
from scipy.sparse import csc_array


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


def test_benchmark_solve_for_steady_state(benchmark):
    """Benchmark solving for steady-state probabilities."""
    state_matrix = csc_array(np.random.rand(101, 100))
    target_vector = np.zeros(101)
    target_vector[-1] = 1.0

    def f():
        solve_for_steady_state(state_matrix, target_vector)

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
