from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Th229-ScanBench benchmark components.")
    parser.add_argument(
        "--baseline",
        choices=["all", "rf"],
        default="all",
        help="`all` runs the full benchmark suite; `rf` runs only the random-forest baseline.",
    )
    args = parser.parse_args()

    if args.baseline == "rf":
        subprocess.run([sys.executable, str(PROJECT_ROOT / "baselines" / "random_forest.py")], check=True)
    else:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "benchmark" / "run_all.py")], check=True)


if __name__ == "__main__":
    main()
