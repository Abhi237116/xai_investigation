import argparse
import json
import os
from pathlib import Path

from financial_crime.data.ieee_cis import write_ieee_cis_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IEEE-CIS Fraud Detection CSV files")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("IEEE_CIS_DATA_DIR"),
        help="Directory containing IEEE-CIS CSV files, or IEEE_CIS_DATA_DIR",
    )
    parser.add_argument(
        "--output",
        default="artifacts/ieee_cis/validation_report.json",
        help="Path for the compact validation report JSON",
    )
    parser.add_argument(
        "--skip-expected-audit",
        action="store_true",
        help="Do not compare against the known full IEEE-CIS audit counts",
    )
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("--data-dir or IEEE_CIS_DATA_DIR is required")

    report = write_ieee_cis_report(
        args.data_dir,
        args.output,
        compare_expected_audit=not args.skip_expected_audit,
    )
    summary = {
        "output": str(Path(args.output)),
        "validation_failures": report["validation_failures"],
        "expected_audit_discrepancies": report.get("expected_audit_discrepancies", {}),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
