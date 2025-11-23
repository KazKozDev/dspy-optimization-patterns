#!/usr/bin/env python3
"""
Data Preparation Script

Converts raw datasets to DSPy format and splits into train/dev/test.

Usage:
    python scripts/prepare_data.py \\
        --input data/raw/qa_dataset_sample.jsonl \\
        --output data/processed/ \\
        --task-type qa
"""

import argparse
import json
from pathlib import Path

from src.pipeline.loader import (
    QADatasetLoader,
    ClassificationDatasetLoader,
    RAGDatasetLoader,
    split_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare data for DSPy optimization")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path (JSONL/JSON/CSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Output directory",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["qa", "classification", "rag"],
        required=True,
        help="Task type",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Training set ratio",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.3,
        help="Dev set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.task_type} dataset from {args.input}...")

    if args.task_type == "qa":
        examples = QADatasetLoader.load(args.input, has_context=True)
    elif args.task_type == "classification":
        examples = ClassificationDatasetLoader.load(args.input)
    elif args.task_type == "rag":
        examples = RAGDatasetLoader.load(args.input)

    # Split dataset
    print(f"\nSplitting dataset ({len(examples)} examples)...")
    trainset, devset, testset = split_dataset(
        examples,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )

    # Save splits
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, dataset in [
        ("train", trainset),
        ("dev", devset),
        ("test", testset),
    ]:
        output_file = output_dir / f"{args.task_type}_{name}.jsonl"
        with open(output_file, "w") as f:
            for example in dataset:
                f.write(json.dumps(example.toDict()) + "\n")
        print(f"✓ Saved {len(dataset)} examples to {output_file}")

    print(f"\n✓ Data preparation complete!")
    print(f"   Total: {len(examples)} examples")
    print(f"   Train: {len(trainset)} ({args.train_ratio:.0%})")
    print(f"   Dev: {len(devset)} ({args.dev_ratio:.0%})")
    print(f"   Test: {len(testset)} ({args.test_ratio:.0%})")


if __name__ == "__main__":
    main()
