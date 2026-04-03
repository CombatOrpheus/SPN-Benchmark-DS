import pytest
import numpy as np
from spn_datasets.generator.PetriGenerate import prune_petri_net, generate_random_petri_net


def test_benchmark_prune_petri_net(benchmark):
    # Setup a matrix
    petri_matrix = generate_random_petri_net(50, 50)

    def f():
        m = petri_matrix.copy()
        prune_petri_net(m)

    benchmark(f)
