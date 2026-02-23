"""
This module provides the DatasetGenerator class for generating SPN datasets.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm, trange

from spn_datasets.generator import DataTransformation as DT
from spn_datasets.generator import PetriGenerate as PeGen
from spn_datasets.generator import SPN


class DatasetGenerator:
    """Generates and augments Stochastic Petri Net (SPN) datasets."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the generator with a configuration.

        Args:
            config: A dictionary containing generation parameters.
        """
        self.config = config

    def generate_single_spn(self) -> Optional[Dict[str, Any]]:
        """Generates a single valid SPN sample.

        Returns:
            A dictionary containing the SPN data, or None if generation fails.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            place_num = np.random.randint(
                self.config["minimum_number_of_places"],
                self.config["maximum_number_of_places"] + 1,
            )
            trans_num = np.random.randint(
                self.config["minimum_number_of_transitions"],
                self.config["maximum_number_of_transitions"] + 1,
            )

            petri_matrix = PeGen.generate_random_petri_net(place_num, trans_num)

            if self.config.get("enable_pruning"):
                petri_matrix = PeGen.prune_petri_net(petri_matrix)

            if self.config.get("enable_token_addition"):
                petri_matrix = PeGen.add_tokens_randomly(petri_matrix)

            results, success = SPN.filter_spn(
                petri_matrix,
                self.config["place_upper_bound"],
                self.config["marks_lower_limit"],
                self.config["marks_upper_limit"],
            )
            if success:
                return results
        return None

    def augment_single_spn(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Augments a single SPN sample.

        Args:
            sample: The original SPN sample.

        Returns:
            A list of augmented samples.
        """
        if not sample or "petri_net" not in sample:
            return []

        petri_net = np.array(sample["petri_net"], dtype="long")

        # Correctly call generate_petri_net_variations with config
        augmented_data = DT.generate_petri_net_variations(petri_net, self.config)

        if not augmented_data:
            return []

        max_transforms = self.config.get(
            "maximum_transformations_per_sample", len(augmented_data)
        )
        if len(augmented_data) > max_transforms:
            indices = np.random.choice(
                len(augmented_data), max_transforms, replace=False
            )
            return [augmented_data[i] for i in indices]
        return augmented_data

    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generates the full dataset based on the configuration.

        Returns:
            A list of all generated (and optionally augmented) samples.
        """
        num_samples = self.config["number_of_samples_to_generate"]
        n_jobs = self.config["number_of_parallel_jobs"]

        print(f"Generating {num_samples} initial SPN samples...")
        initial_samples = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self.generate_single_spn)() for _ in trange(num_samples)
        )
        valid_samples = [s for s in initial_samples if s is not None]
        print(f"Generated {len(valid_samples)} valid initial samples.")

        all_samples = []
        if self.config.get("enable_transformations"):
            print("Augmenting samples...")
            augmented_lists = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(self.augment_single_spn)(sample)
                for sample in tqdm(valid_samples, desc="Augmenting")
            )
            for sample_list in augmented_lists:
                all_samples.extend(sample_list)
        else:
            all_samples = valid_samples

        return all_samples
