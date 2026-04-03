import numpy as np
from spn_datasets.generator.PetriGenerate import generate_random_petri_net
from spn_datasets.generator.SPN import filter_spn
import cProfile
import pstats

matrices = []
for _ in range(500):
    matrices.append(generate_random_petri_net(10, 5))


def run_prof():
    for m in matrices:
        filter_spn(m)


cProfile.run("run_prof()", sort="cumtime")
