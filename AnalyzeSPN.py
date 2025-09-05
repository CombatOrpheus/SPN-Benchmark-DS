import argparse
import h5py
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from utils import DataUtil as DU

def analyze_dataset(hdf5_file_path, output_dir):
    """
    Analyzes the SPN dataset from an HDF5 file.

    Args:
        hdf5_file_path (str): Path to the HDF5 file.
        output_dir (str): Directory to save the analysis results.
    """
    if not os.path.exists(hdf5_file_path):
        print(f"Error: HDF5 file not found at {hdf5_file_path}")
        return

    DU.mkdir(output_dir)

    print(f"Analyzing dataset from {hdf5_file_path}")
    print(f"Output will be saved to {output_dir}")

    all_samples_data = []
    with h5py.File(hdf5_file_path, 'r') as hf:
        dataset_group = hf['dataset_samples']
        for sample_name in dataset_group:
            sample_group = dataset_group[sample_name]
            sample_data = {key: sample_group[key][()] for key in sample_group.keys()}
            all_samples_data.append(sample_data)

    if not all_samples_data:
        print("No data found in the HDF5 file.")
        return

    print(f"Loaded {len(all_samples_data)} samples.")

    # --- Metric Calculation ---
    num_places = [sample['petri_net'].shape[0] for sample in all_samples_data]
    num_transitions = [(sample['petri_net'].shape[1] - 1) // 2 for sample in all_samples_data]
    num_reachable_markings = [len(sample['arr_vlist']) for sample in all_samples_data]
    firing_rates = np.concatenate([sample['spn_labda'] for sample in all_samples_data])
    steady_state_probs = np.concatenate([sample['spn_steadypro'] for sample in all_samples_data])
    avg_tokens_per_place = np.concatenate([sample['spn_allmus'] for sample in all_samples_data])
    total_avg_tokens = [sample['spn_mu'] for sample in all_samples_data]

    # --- Statistical Analysis ---
    stats = {
        "Number of samples": len(all_samples_data),
        "Number of places": {
            "mean": np.mean(num_places),
            "std": np.std(num_places),
            "min": np.min(num_places),
            "max": np.max(num_places),
        },
        "Number of transitions": {
            "mean": np.mean(num_transitions),
            "std": np.std(num_transitions),
            "min": np.min(num_transitions),
            "max": np.max(num_transitions),
        },
        "Number of reachable markings": {
            "mean": np.mean(num_reachable_markings),
            "std": np.std(num_reachable_markings),
            "min": np.min(num_reachable_markings),
            "max": np.max(num_reachable_markings),
        },
        "Firing rates": {
            "mean": np.mean(firing_rates),
            "std": np.std(firing_rates),
            "min": np.min(firing_rates),
            "max": np.max(firing_rates),
        },
        "Steady-state probabilities": {
            "mean": np.mean(steady_state_probs),
            "std": np.std(steady_state_probs),
            "min": np.min(steady_state_probs),
            "max": np.max(steady_state_probs),
        },
        "Average tokens per place": {
            "mean": np.mean(avg_tokens_per_place),
            "std": np.std(avg_tokens_per_place),
            "min": np.min(avg_tokens_per_place),
            "max": np.max(avg_tokens_per_place),
        },
        "Total average tokens": {
            "mean": np.mean(total_avg_tokens),
            "std": np.std(total_avg_tokens),
            "min": np.min(total_avg_tokens),
            "max": np.max(total_avg_tokens),
        },
    }

    # --- Custom JSON encoder to handle numpy types ---
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    # --- Save statistics to a file ---
    stats_file_path = os.path.join(output_dir, "statistics.txt")
    with open(stats_file_path, 'w') as f:
        json.dump(stats, f, indent=4, cls=NpEncoder)
    print(f"Statistics saved to {stats_file_path}")

    # --- Generate and save plots ---
    def save_histogram(data, title, xlabel, filename):
        plt.figure()
        plt.hist(data, bins=30)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    save_histogram(num_places, "Distribution of Number of Places", "Number of Places", "num_places_hist.png")
    save_histogram(num_transitions, "Distribution of Number of Transitions", "Number of Transitions", "num_transitions_hist.png")
    save_histogram(num_reachable_markings, "Distribution of Number of Reachable Markings", "Number of Reachable Markings", "num_reachable_markings_hist.png")
    save_histogram(firing_rates, "Distribution of Firing Rates", "Firing Rate", "firing_rates_hist.png")
    save_histogram(steady_state_probs, "Distribution of Steady-State Probabilities", "Probability", "steady_state_probs_hist.png")
    save_histogram(avg_tokens_per_place, "Distribution of Average Tokens per Place", "Average Tokens", "avg_tokens_per_place_hist.png")
    save_histogram(total_avg_tokens, "Distribution of Total Average Tokens", "Total Average Tokens", "total_avg_tokens_hist.png")

    print("Plots saved to", output_dir)

    # --- Generate and save a summary table (CSV) ---
    import pandas as pd
    df_stats = pd.DataFrame(stats).transpose()
    df_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"))
    print(f"Summary table saved to {os.path.join(output_dir, 'summary_statistics.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SPN dataset.")
    parser.add_argument(
        "--hdf5_file",
        help="Path to the HDF5 file",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save the analysis results",
        default="results/analysis",
    )
    args = parser.parse_args()

    print(f"Attempting to open HDF5 file at: {args.hdf5_file}")
    analyze_dataset(args.hdf5_file, args.output_dir)
    print("Analysis complete.")
