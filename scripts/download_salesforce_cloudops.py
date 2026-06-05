from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Salesforce CloudOps TSF data with HuggingFace datasets.")
    parser.add_argument("--output", default="data/salesforce_cloudops", help="Output directory.")
    parser.add_argument("--subset", default=None, help="Optional dataset config/subset name.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install optional dependency first: python -m pip install datasets") from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset_name = "Salesforce/cloudops_tsf"
    dataset = load_dataset(dataset_name, args.subset) if args.subset else load_dataset(dataset_name)
    dataset.save_to_disk(str(output))
    print(f"Saved {dataset_name} to {output.resolve()}")


if __name__ == "__main__":
    main()

