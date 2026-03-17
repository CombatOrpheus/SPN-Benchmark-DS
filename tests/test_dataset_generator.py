
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from spn_datasets.generator.dataset_generator import DatasetGenerator

class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.config = {
            "minimum_number_of_places": 5,
            "maximum_number_of_places": 10,
            "minimum_number_of_transitions": 5,
            "maximum_number_of_transitions": 10,
            "place_upper_bound": 10,
            "marks_lower_limit": 4,
            "marks_upper_limit": 500,
            "number_of_samples_to_generate": 2,
            "number_of_parallel_jobs": 1,
            "enable_pruning": True,
            "enable_token_addition": True,
            "enable_transformations": True,
            "maximum_transformations_per_sample": 2,
        }
        self.generator = DatasetGenerator(self.config)

    @patch("spn_datasets.generator.dataset_generator.PeGen")
    @patch("spn_datasets.generator.dataset_generator.SPN")
    def test_generate_single_spn(self, mock_spn, mock_pegen):
        # Setup mocks
        mock_pegen.generate_random_petri_net.return_value = np.zeros((5, 11))
        mock_pegen.prune_petri_net.return_value = np.zeros((5, 11))
        mock_pegen.add_tokens_randomly.return_value = np.zeros((5, 11))

        # Mock successful generation
        mock_spn.filter_spn.return_value = ({"petri_net": [[0]]}, True)

        result = self.generator.generate_single_spn()
        self.assertIsNotNone(result)
        self.assertEqual(result, {"petri_net": [[0]]})

        # Mock failure
        mock_spn.filter_spn.return_value = ({}, False)
        result = self.generator.generate_single_spn()
        self.assertIsNone(result)

    @patch("spn_datasets.generator.dataset_generator.DT")
    def test_augment_single_spn(self, mock_dt):
        sample = {"petri_net": np.zeros((5, 11))}

        # Mock transformations
        mock_dt.generate_petri_net_variations.return_value = [{"petri_net": "var1"}, {"petri_net": "var2"}, {"petri_net": "var3"}]

        # Test with limit
        self.config["maximum_transformations_per_sample"] = 2
        results = self.generator.augment_single_spn(sample)
        self.assertEqual(len(results), 2)

        # Test without limit (or larger limit)
        self.config["maximum_transformations_per_sample"] = 5
        results = self.generator.augment_single_spn(sample)
        self.assertEqual(len(results), 3)

        # Test empty
        mock_dt.generate_petri_net_variations.return_value = []
        results = self.generator.augment_single_spn(sample)
        self.assertEqual(len(results), 0)

    @patch("spn_datasets.generator.dataset_generator.Parallel")
    def test_generate_dataset(self, mock_parallel):
        # mock_parallel(...) -> executor
        # executor(...) -> results

        executor_mock = MagicMock()
        mock_parallel.return_value = executor_mock

        # First call: initial samples
        # Second call: augmented samples (if enabled)
        # Note: augment_single_spn returns a LIST of samples.
        # So Parallel returns a list of lists.
        executor_mock.side_effect = [
            [{"sample": 1}, {"sample": 2}], # Initial
            [[{"aug": 1}], [{"aug": 2}]]    # Augmented
        ]

        samples = self.generator.generate_dataset()
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples, [{"aug": 1}, {"aug": 2}])

        # Test without transformations
        self.generator.config["enable_transformations"] = False
        executor_mock.side_effect = [
            [{"sample": 1}, {"sample": 2}]
        ]
        samples = self.generator.generate_dataset()
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples, [{"sample": 1}, {"sample": 2}])
