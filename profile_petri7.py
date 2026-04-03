import numpy as np
from spn_datasets.generator.PetriGenerate import generate_random_petri_net
import cProfile

cProfile.run("for _ in range(5000): generate_random_petri_net(10, 5)", sort="cumtime")
