import argparse
import json
import os
from pathlib import Path

from financial_crime.data.ieee_cis import load_ieee_cis_dataset
from financial_crime.ml.train import train_ieee_cis_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark IEEE-CIS baseline models")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("IEEE_CIS_DATA_DIR"),
        help="Directory containing the IEEE-CIS CSV files, or IEEE_CIS_DATA_DIR",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/phase3_artifacts",
        help="Directory where benchmark models, metrics, and plots are saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for splits and model initialization",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Validation share for the held-out split",
    )
    parser.add_argument(
        "--skip-expected-audit",
        action="store_true",
        help="Skip validation against the known full IEEE-CIS audit values",
    )
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("--data-dir or IEEE_CIS_DATA_DIR is required")

    dataset = load_ieee_cis_dataset(
        args.data_dir,
        compare_expected_audit=not args.skip_expected_audit,
    )
    summary = train_ieee_cis_baselines(
        dataset,
        args.output_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    summary_text = {
        "output_dir": str(Path(args.output_dir)),
        "split_strategy": summary["dataset"]["split_strategy"],
        "train_rows": summary["dataset"]["train_rows"],
        "validation_rows": summary["dataset"]["validation_rows"],
        "models": {
            name: {
                "metrics": payload["metrics"],
                "artifact_path": payload["artifact_path"],
                "predictions_path": payload["predictions_path"],
            }
            for name, payload in summary["models"].items()
        },
    }
    print(json.dumps(summary_text, indent=2))


if __name__ == "__main__":
    main()
