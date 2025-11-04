import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    # Ensure we are in the repo root
    repo_root = Path(__file__).parent.parent.resolve()
    os.chdir(repo_root)

    parser = argparse.ArgumentParser(description="Benchmark different commits.")
    parser.add_argument("commits", nargs="+", help="List of commit hashes to benchmark.")
    parser.add_argument(
        "--output-csv",
        default="benchmark_results.csv",
        help="Path to save the benchmark results in CSV format.",
    )
    args = parser.parse_args()

    results = []
    original_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()

    try:
        for commit in args.commits:
            print(f"--- Benchmarking commit {commit} ---")
            results.append(benchmark_commit(commit, original_branch))
    finally:
        print(f"--- Restoring original branch: {original_branch} ---")
        # Use --force to discard any changes made by the benchmarked code
        subprocess.run(["git", "checkout", "--force", original_branch], check=True)

    if not results:
        print("No benchmarks were run.")
        return

    # Save results to CSV
    print(f"--- Saving results to {args.output_csv} ---")
    fieldnames = results[0].keys()
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print results to console
    print("\n--- Benchmark Results ---")
    header = f"{'Commit':<10} | {'Data Gen Time (s)':<20}"
    print(header)
    print("-" * len(header))
    for result in results:
        if "error" in result:
            print(f"{result['commit']:<10} | {'ERROR':<20}")
            print(f"  Error: {result['error']}")
        else:
            print(f"{result['commit'][:7]:<10} | " f"{result['data_gen_time']:<20.4f}")


def run_command(command, error_message):
    """Runs a command and raises an exception if it fails."""
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"{error_message}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise e


def benchmark_commit(commit, original_branch):
    """Checks out a commit, installs dependencies, and runs benchmarks."""
    commit_results = {"commit": commit}

    try:
        # Checkout the commit and clean the directory
        run_command(["git", "checkout", commit], f"Failed to checkout commit {commit}.")
        run_command(["git", "clean", "-fdx"], "Failed to clean the working directory.")

        print("--- Installing dependencies ---")
        run_command(["uv", "sync", "--all-extras"], "Failed to install dependencies.")
        run_command(["uv", "pip", "install", "-e", "."], "Failed to install project in editable mode.")

        print("--- Benchmarking data generation ---")
        commit_results["data_gen_time"] = benchmark_data_generation()

    except Exception as e:
        print(f"An error occurred while benchmarking commit {commit}: {e}")
        # Ensure we can continue to the next commit
        commit_results["error"] = str(e)
    finally:
        # Always checkout the original branch to be able to switch to the next commit
        # Use --force to discard any changes.
        subprocess.run(["git", "checkout", "--force", original_branch], check=True)

    return commit_results


def benchmark_data_generation():
    """Runs the data generation script and returns the execution time."""
    config_file = Path("benchmarks/benchmark.toml")
    if not config_file.exists():
        print(f"Error: Data generation config file not found at {config_file}")
        return -1

    command = ["uv", "run", "python", "SPNGenerate.py", "--config", str(config_file)]

    start_time = time.time()
    run_command(command, "Data generation script failed.")
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    main()
