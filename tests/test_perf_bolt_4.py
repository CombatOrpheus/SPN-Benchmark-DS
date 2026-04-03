import pytest
import numpy as np
from spn_datasets.generator.SPN import is_connected, compute_qualitative_properties


def test_benchmark_is_connected(benchmark):
    petri_matrix = np.random.randint(0, 2, size=(50, 101)).astype(np.int32)

    def f():
        is_connected(petri_matrix)

    benchmark(f)


def test_benchmark_compute_qualitative_properties(benchmark):
    vertices = [np.random.randint(0, 5, size=(50,)).astype(np.int32) for _ in range(500)]
    edges = [[i, (i + 1) % 500] for i in range(500)]

    def f():
        compute_qualitative_properties(vertices, edges)

    benchmark(f)
