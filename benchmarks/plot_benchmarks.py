import argparse
import csv
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "csv_file",
        help="Path to the benchmark results CSV file.",
    )
    parser.add_argument(
        "--output-image",
        default="benchmark_plot.png",
        help="Path to save the plot image.",
    )
    args = parser.parse_args()

    results = []
    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    if not results:
        print("No results to plot.")
        return

    # Filter out errored commits
    valid_results = [r for r in results if "error" not in r and r.get("data_gen_time")]
    if not valid_results:
        print("No valid results to plot.")
        return

    commits = [r["commit"][:7] for r in valid_results]
    times = [float(r["data_gen_time"]) for r in valid_results]

    plt.figure(figsize=(10, 6))
    plt.bar(commits, times, color="skyblue")
    plt.xlabel("Commit")
    plt.ylabel("Data Generation Time (s)")
    plt.title("Benchmark: Data Generation Time per Commit")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    print(f"--- Saving plot to {args.output_image} ---")
    plt.savefig(args.output_image)
    plt.close()
    print(f"Plot saved to {args.output_image}")


if __name__ == "__main__":
    main()
