#!/usr/bin/env python
"""
This script profiles SPNGenerate.py using pyinstrument and outputs an interactive HTML flamegraph.
It passes any additional command line arguments through to SPNGenerate.py.
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Setup the profiling directory
    profiling_dir = Path("profiling")
    profiling_dir.mkdir(exist_ok=True)

    # Configure the output file path for pyinstrument
    output_file = profiling_dir / "flamegraph.html"

    cmd = [
        "uv", "run", "pyinstrument",
        "-r", "html",
        "-o", str(output_file),
        "SPNGenerate.py"
    ]

    # Append any arguments passed to this script
    cmd.extend(sys.argv[1:])

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run pyinstrument using subprocess.
        subprocess.run(cmd, check=True)
        print(f"\nProfiling complete! Interactive HTML flamegraph saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"\nError running pyinstrument: {e}", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\nError: Could not find 'uv'. Ensure it is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
