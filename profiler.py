#!/usr/bin/env python
"""
This script profiles SPNGenerate.py using py-spy and outputs a flamegraph.
It passes any additional command line arguments through to SPNGenerate.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    # Setup the profiling directory
    profiling_dir = Path("profiling")
    profiling_dir.mkdir(exist_ok=True)

    # Configure the output file path for py-spy
    output_file = profiling_dir / "flamegraph.svg"

    cmd = [
        "uv", "run", "py-spy", "record",
        "-o", str(output_file),
        "-f", "flamegraph",
        "--",
        "python", "SPNGenerate.py"
    ]

    # Append any arguments passed to this script
    cmd.extend(sys.argv[1:])

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run py-spy using subprocess.
        # Note: py-spy may need sudo/admin rights to profile processes on some systems.
        # With subprocesses, using `--subprocesses` flag in py-spy could be useful, but let's stick to default.
        subprocess.run(cmd, check=False)
        # py-spy currently returns non-zero exit code due to No child process (os error 10) on some container environments
        # when the profiled process finishes execution. We check if output_file was created to determine success.

        if output_file.exists():
            print(f"\nProfiling complete! Flamegraph saved to: {output_file}")
            sys.exit(0)
        else:
            print("\nError: py-spy did not produce an output file.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"\nError running py-spy: {e}", file=sys.stderr)
        print("Note: On Linux, you might need to run this script with 'sudo' or set ptrace_scope.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
