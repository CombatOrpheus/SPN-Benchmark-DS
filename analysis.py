import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def extract_data_from_hdf5(dataset_path):
    """
    Extracts metrics from the SPN HDF5 dataset.

    Args:
        dataset_path (str): Path to the HDF5 file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted metrics for each sample.
        dict: A dictionary containing lists of vector data for global analysis.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return None, None

    scalar_metrics = []
    vector_metrics = {
        'spn_labda': [],
        'spn_steadypro': [],
        'spn_allmus': [],
    }

    with h5py.File(dataset_path, 'r') as hf:
        if 'dataset_samples' not in hf:
            print("Error: 'dataset_samples' group not found in HDF5 file.")
            return None, None

        sample_groups = sorted(hf['dataset_samples'].keys())
        print(f"Found {len(sample_groups)} samples. Starting data extraction...")

        for sample_name in tqdm(sample_groups, desc="Extracting Samples"):
            sample_group = hf['dataset_samples'][sample_name]

            try:
                # --- Scalar metrics ---
                spn_mu = sample_group['spn_mu'][()]

                # --- Derived scalar metrics ---
                petri_net = sample_group['petri_net'][()]
                num_places = petri_net.shape[0]
                num_transitions = (petri_net.shape[1] - 1) // 2 if petri_net.ndim > 1 else 0

                arr_vlist = sample_group['arr_vlist'][()]
                num_states = arr_vlist.shape[0]

                arr_edge = sample_group['arr_edge'][()]
                num_edges = arr_edge.shape[0]

                # --- Vector metrics ---
                labda = sample_group['spn_labda'][()]
                steadypro = sample_group['spn_steadypro'][()]
                allmus = sample_group['spn_allmus'][()]

                scalar_metrics.append({
                    'sample_name': sample_name,
                    'spn_mu': spn_mu,
                    'num_places': num_places,
                    'num_transitions': num_transitions,
                    'num_states': num_states,
                    'num_edges': num_edges,
                    'labda_mean': np.mean(labda) if labda.size > 0 else 0,
                    'steadypro_mean': np.mean(steadypro) if steadypro.size > 0 else 0,
                    'allmus_mean': np.mean(allmus) if allmus.size > 0 else 0,
                })

                # Append full vectors for global analysis
                vector_metrics['spn_labda'].extend(labda)
                vector_metrics['spn_steadypro'].extend(steadypro)
                vector_metrics['spn_allmus'].extend(allmus)

            except KeyError as e:
                print(f"Warning: Skipping sample {sample_name} due to missing key: {e}")
            except Exception as e:
                print(f"Warning: Skipping sample {sample_name} due to an error: {e}")

    df = pd.DataFrame(scalar_metrics)
    if 'sample_name' in df.columns:
        df = df.set_index('sample_name')

    return df, vector_metrics


def generate_summary_tables(scalar_df, vector_data, output_folder):
    """
    Generates statistical summary tables and saves them to CSV files.

    Args:
        scalar_df (pd.DataFrame): DataFrame with scalar metrics per sample.
        vector_data (dict): Dictionary with lists of all vector elements.
        output_folder (str): Path to the folder to save the CSV files.
    """
    print("\nGenerating summary tables...")

    # 1. Summary for scalar metrics
    scalar_summary = scalar_df.describe()
    scalar_summary_path = os.path.join(output_folder, "scalar_metrics_summary.csv")
    scalar_summary.to_csv(scalar_summary_path)
    print(f"Saved scalar metrics summary to: {scalar_summary_path}")

    # 2. Summary for global vector metrics
    vector_summaries = {}
    for name, data in vector_data.items():
        if data:
            s = pd.Series(data)
            vector_summaries[name] = s.describe()

    if vector_summaries:
        vector_summary_df = pd.DataFrame(vector_summaries)
        vector_summary_path = os.path.join(output_folder, "vector_metrics_summary.csv")
        vector_summary_df.to_csv(vector_summary_path)
        print(f"Saved vector metrics summary to: {vector_summary_path}")

    print("\nTable generation complete.")


def generate_plots(scalar_df, vector_data, output_folder):
    """
    Generates and saves plots for the analyzed metrics.

    Args:
        scalar_df (pd.DataFrame): DataFrame with scalar metrics per sample.
        vector_data (dict): Dictionary with lists of all vector elements.
        output_folder (str): Path to the folder to save the plots.
    """
    print("\nGenerating plots...")
    plt.style.use('ggplot')

    # 1. Histograms for scalar metrics
    for column in scalar_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(scalar_df[column], bins=50, alpha=0.7)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plot_path = os.path.join(output_folder, f'hist_{column}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved histogram to: {plot_path}")

    # 2. Histograms for global vector metrics
    for name, data in vector_data.items():
        if data:
            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=50, alpha=0.7, label=name)
            plt.title(f'Distribution of all {name} elements')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plot_path = os.path.join(output_folder, f'hist_all_{name}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved histogram to: {plot_path}")

    # 3. Scatter plots for interesting relationships
    scatter_pairs = [
        ('num_places', 'spn_mu'),
        ('num_transitions', 'spn_mu'),
        ('num_states', 'spn_mu'),
        ('num_places', 'num_transitions')
    ]
    for x_col, y_col in scatter_pairs:
        if x_col in scalar_df.columns and y_col in scalar_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(scalar_df[x_col], scalar_df[y_col], alpha=0.5)
            plt.title(f'{x_col} vs. {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True)
            plot_path = os.path.join(output_folder, f'scatter_{x_col}_vs_{y_col}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved scatter plot to: {plot_path}")

    print("\nPlot generation complete.")


def main():
    """
    Main function to parse arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze an SPN dataset from an HDF5 file."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the HDF5 dataset file.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help="Folder to save the analysis results (default: results).",
    )
    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}")

    # --- Data Extraction ---
    scalar_df, vector_data = extract_data_from_hdf5(args.dataset_path)

    if scalar_df is not None and not scalar_df.empty:
        print("\nData extraction complete.")
        print(f"Successfully processed {len(scalar_df)} samples.")

        # --- Analysis and Table Generation ---
        generate_summary_tables(scalar_df, vector_data, args.output_folder)

        # --- Plot Generation ---
        generate_plots(scalar_df, vector_data, args.output_folder)

        print("\nAnalysis finished successfully.")
    else:
        print("\nData extraction failed or returned no data. Exiting.")


if __name__ == "__main__":
    main()
