import time
from spn_datasets.generator.dataset_generator import DatasetGenerator

config = {
    "minimum_number_of_places": 5,
    "maximum_number_of_places": 10,
    "minimum_number_of_transitions": 5,
    "maximum_number_of_transitions": 10,
    "place_upper_bound": 10,
    "marks_lower_limit": 4,
    "marks_upper_limit": 500,
    "enable_pruning": True,
    "enable_token_addition": True,
    "number_of_samples_to_generate": 100,
    "number_of_parallel_jobs": 1,
    "enable_transformations": True,
    "enable_delete_edge": True,
    "enable_add_edge": True,
    "enable_add_token": True,
    "enable_delete_token": True,
    "enable_add_place": True,
    "max_candidates_per_structure": 10,
    "maximum_transformations_per_sample": 5,
    "enable_rate_variations": True,
    "num_rate_variations_per_structure": 2
}

gen = DatasetGenerator(config)

t0 = time.perf_counter()
for _ in range(3):
    gen.generate_dataset()
t1 = time.perf_counter()
print(f"generate_dataset: {t1 - t0:.4f}s")
