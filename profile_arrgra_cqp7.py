import numpy as np
import time
from spn_datasets.generator.SPN import filter_spn
import spn_datasets.generator.PetriGenerate as PeGen

matrices = [PeGen.generate_random_petri_net(10, 5) for _ in range(500)]

# Warmup
filter_spn(matrices[0])

import cProfile

cProfile.run("for m in matrices: filter_spn(m)", sort="cumtime")
