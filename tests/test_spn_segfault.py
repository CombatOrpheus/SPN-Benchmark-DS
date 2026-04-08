import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from spn_datasets.generator.dataset_generator import DatasetGenerator

def test_spsolve_segfault_regression():
    """
    Test that large-scale generation does not trigger segmentation faults
    when computing the steady state.
    """
    config = {
        "minimum_number_of_places": 10,
        "maximum_number_of_places": 30,
        "minimum_number_of_transitions": 10,
        "maximum_number_of_transitions": 30,
        "place_upper_bound": 20,
        "marks_lower_limit": 4,
        "marks_upper_limit": 500,
        "enable_pruning": True,
        "enable_token_addition": True,
        "number_of_samples_to_generate": 1000,
        "number_of_parallel_jobs": 4,
        "enable_transformations": False
    }

    gen = DatasetGenerator(config)

    try:
        samples = gen.generate_dataset()
        assert len(samples) == 1000, f"Expected 1000 samples, got {len(samples)}"
    except Exception as e:
        pytest.fail(f"Dataset generation failed: {e}")
