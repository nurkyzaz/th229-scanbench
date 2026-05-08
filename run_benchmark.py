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
        choices=["all", "rf", "sbi_npe", "neural_cnn", "neural_transformer"],
        default="all",
        help="`all` runs the full benchmark suite; ML baselines can also be run individually.",
    )
    parser.add_argument("--sbi-num-simulations", type=int, default=None, help="Override SBI simulator draws.")
    parser.add_argument("--sbi-max-epochs", type=int, default=None, help="Override SBI training epochs.")
    parser.add_argument("--sbi-posterior-samples", type=int, default=None, help="Override SBI posterior samples per example.")
    parser.add_argument("--sbi-bootstrap", type=int, default=None, help="Override SBI A95 bootstrap replicates.")
    parser.add_argument("--sbi-checkpoint", type=Path, default=None, help="Override SBI checkpoint path.")
    parser.add_argument("--sbi-output-tag", default=None, help="Override SBI output artifact prefix.")
    parser.add_argument("--sbi-simulator-null", choices=["parametric", "flow"], default=None, help="Override SBI simulator null.")
    parser.add_argument("--neural-max-epochs", type=int, default=None, help="Override supervised neural training epochs.")
    parser.add_argument("--neural-patience", type=int, default=None, help="Override supervised neural early-stopping patience.")
    parser.add_argument("--neural-bootstrap", type=int, default=None, help="Override supervised neural A95 bootstrap replicates.")
    parser.add_argument("--force-train", action="store_true", help="Retrain an existing SBI checkpoint.")
    args = parser.parse_args()

    if args.baseline == "rf":
        subprocess.run([sys.executable, str(PROJECT_ROOT / "baselines" / "random_forest.py")], check=True)
    elif args.baseline == "sbi_npe":
        command = [sys.executable, str(PROJECT_ROOT / "baselines" / "sbi_npe.py")]
        if args.sbi_num_simulations is not None:
            command.extend(["--num-simulations", str(args.sbi_num_simulations)])
        if args.sbi_max_epochs is not None:
            command.extend(["--max-epochs", str(args.sbi_max_epochs)])
        if args.sbi_posterior_samples is not None:
            command.extend(["--posterior-samples", str(args.sbi_posterior_samples)])
        if args.sbi_bootstrap is not None:
            command.extend(["--bootstrap", str(args.sbi_bootstrap)])
        if args.sbi_checkpoint is not None:
            command.extend(["--checkpoint", str(args.sbi_checkpoint)])
        if args.sbi_output_tag is not None:
            command.extend(["--output-tag", args.sbi_output_tag])
        if args.sbi_simulator_null is not None:
            command.extend(["--simulator-null", args.sbi_simulator_null])
        if args.force_train:
            command.append("--force-train")
        subprocess.run(command, check=True)
    elif args.baseline in {"neural_cnn", "neural_transformer"}:
        module = "neural_cnn.py" if args.baseline == "neural_cnn" else "neural_transformer.py"
        command = [sys.executable, str(PROJECT_ROOT / "baselines" / module)]
        if args.neural_max_epochs is not None:
            command.extend(["--max-epochs", str(args.neural_max_epochs)])
        if args.neural_patience is not None:
            command.extend(["--patience", str(args.neural_patience)])
        if args.neural_bootstrap is not None:
            command.extend(["--bootstrap", str(args.neural_bootstrap)])
        subprocess.run(command, check=True)
    else:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "benchmark" / "run_all.py")], check=True)


if __name__ == "__main__":
    main()
