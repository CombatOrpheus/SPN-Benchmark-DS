"""
This script generates a statistical report from an SPN dataset file.
The report includes plots and tables summarizing the dataset's characteristics
and the configuration used to generate it. The final output is an HTML file.
"""

import argparse
from pathlib import Path
import json
import base64
from io import BytesIO

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_arg_parser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate a statistical report from an SPN dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset file (HDF5 or JSONL).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output HTML report.",
    )
    return parser


def load_data(filepath):
    """Loads data from HDF5 or JSONL and extracts key statistics."""
    filepath = Path(filepath)
    file_ext = filepath.suffix
    stats_list = []
    config = {}

    print(f"Loading data from {filepath}...")
    try:
        if file_ext == ".hdf5":
            with h5py.File(filepath, "r") as hf:
                if "generation_config" in hf.attrs:
                    config = json.loads(hf.attrs["generation_config"])

                dataset_group = hf.get("dataset_samples")
                if dataset_group:
                    for sample_name in dataset_group:
                        sample = dataset_group[sample_name]
                        petri_net = sample["petri_net"][:]
                        arr_vlist = sample["arr_vlist"][:]
                        stats_list.append(
                            {
                                "places": petri_net.shape[0],
                                "transitions": (petri_net.shape[1] - 1) // 2,
                                "states": len(arr_vlist),
                            }
                        )

        elif file_ext == ".jsonl":
            with open(filepath, "r") as f:
                config_line = f.readline()
                if config_line:
                    config = json.loads(config_line)

                for line in f:
                    sample = json.loads(line)
                    petri_net = np.array(sample["petri_net"])
                    arr_vlist = np.array(sample["arr_vlist"])
                    stats_list.append(
                        {
                            "places": petri_net.shape[0],
                            "transitions": (petri_net.shape[1] - 1) // 2,
                            "states": len(arr_vlist),
                        }
                    )
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), {}

    print(f"Successfully loaded {len(stats_list)} samples.")
    return pd.DataFrame(stats_list), config


def main():
    """Main function to generate the statistical report."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    stats_df, config = load_data(input_path)

    if stats_df.empty:
        print("No data loaded. Exiting.")
        return

    if stats_df.empty:
        print("No data loaded. Exiting.")
        return

    # Generate all the components for the report
    plots = generate_plots(stats_df)
    config_table_html = create_config_table(config)

    # Generate all the components for the report
    plots = generate_plots(stats_df)
    config_table_html = create_config_table(config)

    # Assemble the final HTML report
    generate_html_report(plots, config_table_html, args.output)

    print(f"Successfully generated report at: {args.output}")


def generate_plots(stats_df):
    """Generates plots from the statistics DataFrame."""
    plots = {}
    sns.set_theme(style="whitegrid")

    # Plot 1: Distribution of Places
    plt.figure(figsize=(10, 6))
    sns.histplot(data=stats_df, x="places", kde=True)
    plt.title("Distribution of Places")
    plots["places_dist"] = _plot_to_base64(plt)

    # Plot 2: Distribution of Transitions
    plt.figure(figsize=(10, 6))
    sns.histplot(data=stats_df, x="transitions", kde=True)
    plt.title("Distribution of Transitions")
    plots["transitions_dist"] = _plot_to_base64(plt)

    # Plot 3: Distribution of States
    plt.figure(figsize=(10, 6))
    sns.histplot(data=stats_df, x="states", kde=True)
    plt.title("Distribution of States")
    plots["states_dist"] = _plot_to_base64(plt)

    # Plot 4: Places vs. Transitions
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=stats_df, x="places", y="transitions", hue="states", size="states", sizes=(20, 200), palette="viridis"
    )
    plt.title("Places vs. Transitions (colored by number of states)")
    plots["places_vs_transitions"] = _plot_to_base64(plt)

    print(f"Generated {len(plots)} plots.")
    return plots


def _plot_to_base64(plt_obj):
    """Converts a matplotlib plot to a Base64 encoded string."""
    img_buffer = BytesIO()
    plt_obj.savefig(img_buffer, format="png", bbox_inches="tight")
    plt_obj.close()
    img_buffer.seek(0)
    encoded_string = base64.b64encode(img_buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"


def create_config_table(config):
    """Creates an HTML table from the configuration dictionary."""
    if not config:
        return "<p>No configuration data found.</p>"

    config_df = pd.DataFrame(list(config.items()), columns=["Parameter", "Value"])
    html_table = config_df.to_html(index=False, border=0, classes="table table-striped mt-4")
    print("Generated configuration table.")
    return html_table


def generate_html_report(plots, config_table_html, output_path):
    """Generates a self-contained HTML report from the plots and config table."""

    plots_html = ""
    for title, base64_img in plots.items():
        plots_html += f"""
        <div class="col-md-6 mb-4">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title text-center">{title.replace("_", " ").title()}</h5>
                    <img src="{base64_img}" class="img-fluid" alt="{title}">
                </div>
            </div>
        </div>
        """

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SPN Dataset Statistical Report</title>
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-color: #f8f9fa;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}
            .container {{
                max-width: 1200px;
            }}
            .card {{
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                transition: 0.3s;
            }}
            .card:hover {{
                box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
            }}
            .table-striped tbody tr:nth-of-type(odd) {{
                background-color: rgba(0,0,0,.05);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">SPN Dataset Statistical Report</h1>

            <h2 class="mt-5 mb-3">Generation Configuration</h2>
            {config_table_html}

            <h2 class="mt-5 mb-3">Dataset Statistics</h2>
            <div class="row">
                {plots_html}
            </div>
        </div>
    </body>
    </html>
    """

    try:
        with open(output_path, "w") as f:
            f.write(html_template)
    except IOError as e:
        print(f"Error writing HTML report to {output_path}: {e}")


if __name__ == "__main__":
    main()
