import time
from spn_datasets.generator.PetriGenerate import generate_random_petri_net
from spn_datasets.generator.DataTransformation import generate_petri_net_variations

# warmup
petri_matrix = generate_random_petri_net(30, 20)
config = {
    "enable_add_edge": True,
    "enable_delete_edge": True,
    "enable_add_place": True,
    "enable_add_token": True,
    "enable_delete_token": True,
    "enable_rate_variations": True,
    "number_of_parallel_jobs": 1,
}
generate_petri_net_variations(petri_matrix, config)

start = time.time()
generate_petri_net_variations(petri_matrix, config)
print("Time parallel=1:", time.time() - start)

config["number_of_parallel_jobs"] = -1
generate_petri_net_variations(petri_matrix, config)
start = time.time()
generate_petri_net_variations(petri_matrix, config)
print("Time parallel=-1:", time.time() - start)
