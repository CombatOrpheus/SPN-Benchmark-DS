import numpy as np
import pytest
from spn_datasets.generator import RuleBasedPetriNetGenerator, RandomPetriNetGenerator
from spn_datasets.generator.SPN import is_connected

def test_rule_based_generator_structure():
    """Test that RuleBasedPetriNetGenerator produces correct structure."""
    num_places = 10
    num_transitions = 8
    generator = RuleBasedPetriNetGenerator()
    petri_matrix = generator.generate(num_places, num_transitions)

    assert petri_matrix.shape == (num_places, 2 * num_transitions + 1)

    # Check if connected
    assert is_connected(petri_matrix)

    # Check if initial marking exists
    assert np.sum(petri_matrix[:, -1]) > 0

def test_rule_based_generator_small():
    """Test with small size."""
    num_places = 2
    num_transitions = 2
    generator = RuleBasedPetriNetGenerator()
    petri_matrix = generator.generate(num_places, num_transitions)

    assert petri_matrix.shape == (num_places, 2 * num_transitions + 1)
    assert is_connected(petri_matrix)

def test_random_generator_structure():
    """Test that RandomPetriNetGenerator works as expected (sanity check)."""
    num_places = 5
    num_transitions = 5
    generator = RandomPetriNetGenerator()
    petri_matrix = generator.generate(num_places, num_transitions)
    assert petri_matrix.shape == (num_places, 2 * num_transitions + 1)
