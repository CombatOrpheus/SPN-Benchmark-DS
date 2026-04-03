import pytest
import numpy as np
from spn_datasets.generator.PetriGenerate import add_missing_connections, delete_excess_edges


def test_benchmark_add_missing_connections_original(benchmark):
    petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)
    # Zero out some to trigger the missing connections
    petri_matrix[:, 5:10] = 0
    petri_matrix[20:25, :] = 0

    def f():
        m = petri_matrix.copy()
        add_missing_connections(m, 50)

    benchmark(f)


def test_benchmark_delete_excess_edges_original(benchmark):
    petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)

    def f():
        m = petri_matrix.copy()
        delete_excess_edges(m, 50)

    benchmark(f)
