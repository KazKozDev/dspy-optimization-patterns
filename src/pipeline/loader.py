"""
Data Loading and Preparation for DSPy

DSPy expects data as List[dspy.Example].
Each Example contains:
- Input fields (question, context, etc.)
- Output fields (answer, category, etc.)

Best Practices:
1. Always split data: Train / Dev / Test
2. Train: Used by optimizer to generate few-shot examples
3. Dev: Used during optimization to evaluate candidates
4. Test: Hold-out set for final evaluation (NEVER seen by optimizer)
5. Keep Test set representative of production data
"""

import json
import random
from pathlib import Path
from typing import Any

import dspy

# ============================================================================
# DATA LOADING FROM COMMON FORMATS
# ============================================================================


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load data from JSONL file (one JSON object per line)."""
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(file_path: str) -> list[dict[str, Any]]:
    """Load data from JSON file (array of objects)."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_csv(file_path: str) -> list[dict[str, Any]]:
    """Load data from CSV file."""
    import csv

    data = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data


# ============================================================================
# CONVERSION TO DSPY.EXAMPLE
# ============================================================================


def dict_to_example(
    data_dict: dict[str, Any],
    input_keys: list[str],
    output_keys: list[str],
) -> dspy.Example:
    """
    Convert dictionary to dspy.Example.

    Args:
        data_dict: Raw data dictionary
        input_keys: Keys to use as inputs (e.g., ["question", "context"])
        output_keys: Keys to use as outputs (e.g., ["answer"])

    Returns:
        dspy.Example with inputs and outputs properly labeled
    """
    # Build Example with all fields
    example_data = {}

    for key in input_keys:
        if key in data_dict:
            example_data[key] = data_dict[key]

    for key in output_keys:
        if key in data_dict:
            example_data[key] = data_dict[key]

    # Create Example and mark which fields are inputs
    example = dspy.Example(**example_data).with_inputs(*input_keys)

    return example


# ============================================================================
# DATASET SPLITTING
# ============================================================================


def split_dataset(
    examples: list[dspy.Example],
    train_ratio: float = 0.5,
    dev_ratio: float = 0.3,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """
    Split dataset into train/dev/test sets.

    Args:
        examples: List of dspy.Example objects
        train_ratio: Proportion for training (optimizer uses this)
        dev_ratio: Proportion for validation during optimization
        test_ratio: Proportion for final hold-out evaluation
        random_seed: For reproducibility

    Returns:
        (trainset, devset, testset)
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"

    random.seed(random_seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    trainset = shuffled[:train_end]
    devset = shuffled[train_end:dev_end]
    testset = shuffled[dev_end:]

    print(f"✓ Split dataset: Train={len(trainset)}, Dev={len(devset)}, Test={len(testset)}")

    return trainset, devset, testset


# ============================================================================
# DATASET LOADERS FOR COMMON TASKS
# ============================================================================


class QADatasetLoader:
    """
    Load Question-Answering datasets.

    Expected format:
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": "France is a country... Paris is the capital."  # Optional
    }
    """

    @staticmethod
    def load(file_path: str, has_context: bool = False) -> list[dspy.Example]:
        """Load QA dataset from file."""
        path = Path(file_path)

        if path.suffix == ".jsonl":
            data = load_jsonl(file_path)
        elif path.suffix == ".json":
            data = load_json(file_path)
        elif path.suffix == ".csv":
            data = load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Convert to Examples
        input_keys = ["question", "context"] if has_context else ["question"]
        output_keys = ["answer"]

        examples = [
            dict_to_example(item, input_keys, output_keys)
            for item in data
            if "question" in item and "answer" in item
        ]

        print(f"✓ Loaded {len(examples)} QA examples from {file_path}")
        return examples


class ClassificationDatasetLoader:
    """
    Load classification datasets.

    Expected format:
    {
        "text": "This product is amazing!",
        "category": "positive"
    }
    """

    @staticmethod
    def load(
        file_path: str, text_key: str = "text", label_key: str = "category"
    ) -> list[dspy.Example]:
        """Load classification dataset."""
        path = Path(file_path)

        if path.suffix == ".jsonl":
            data = load_jsonl(file_path)
        elif path.suffix == ".json":
            data = load_json(file_path)
        elif path.suffix == ".csv":
            data = load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        examples = [
            dict_to_example(
                item,
                input_keys=[text_key],
                output_keys=[label_key],
            )
            for item in data
            if text_key in item and label_key in item
        ]

        print(f"✓ Loaded {len(examples)} classification examples from {file_path}")
        return examples


class RAGDatasetLoader:
    """
    Load RAG (Retrieval-Augmented Generation) datasets.

    Expected format:
    {
        "question": "When was the first iPhone released?",
        "answer": "2007",
        "contexts": [
            "The iPhone was released in 2007...",
            "Apple announced the iPhone in January 2007..."
        ]
    }
    """

    @staticmethod
    def load(file_path: str) -> list[dspy.Example]:
        """Load RAG dataset."""
        path = Path(file_path)

        if path.suffix == ".jsonl":
            data = load_jsonl(file_path)
        elif path.suffix == ".json":
            data = load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        examples = []
        for item in data:
            if "question" not in item or "answer" not in item:
                continue

            # Handle contexts field
            contexts = item.get("contexts", [])
            if isinstance(contexts, str):
                contexts = [contexts]

            example = dspy.Example(
                question=item["question"],
                answer=item["answer"],
                contexts=contexts,
            ).with_inputs("question")

            examples.append(example)

        print(f"✓ Loaded {len(examples)} RAG examples from {file_path}")
        return examples


# ============================================================================
# DATASET AUGMENTATION
# ============================================================================


def augment_with_negatives(
    examples: list[dspy.Example],
    num_negatives: int = 2,
) -> list[dspy.Example]:
    """
    Augment dataset with negative examples.

    For classification: Add examples with wrong labels
    For QA: Add questions with wrong answers

    Helps optimizer learn to reject bad outputs.
    """
    augmented = examples.copy()

    for example in examples[:num_negatives]:
        # Create negative by swapping answer with another example's answer
        other = random.choice(examples)
        if hasattr(example, "answer") and hasattr(other, "answer"):
            negative = example.copy()
            negative.answer = other.answer
            augmented.append(negative)

    print(f"✓ Augmented dataset: {len(examples)} -> {len(augmented)} examples")
    return augmented


# ============================================================================
# DATASET VALIDATION
# ============================================================================


def validate_dataset(examples: list[dspy.Example], required_fields: list[str]):
    """
    Validate that all examples have required fields.

    Raises ValueError if any example is missing fields.
    """
    for i, example in enumerate(examples):
        missing_fields = [field for field in required_fields if not hasattr(example, field)]

        if missing_fields:
            raise ValueError(
                f"Example {i} missing fields: {missing_fields}\n" f"Example: {example}"
            )

    print(f"✓ Validated {len(examples)} examples with fields: {required_fields}")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def load_and_split(
    file_path: str,
    task_type: str = "qa",
    train_size: int = 50,
    dev_size: int = 100,
    test_size: int = 200,
    random_seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """
    One-stop function to load and split dataset.

    Args:
        file_path: Path to data file
        task_type: "qa", "classification", or "rag"
        train_size: Number of examples for training
        dev_size: Number for validation
        test_size: Number for testing
        random_seed: For reproducibility

    Returns:
        (trainset, devset, testset)
    """
    # Load dataset based on task type
    if task_type == "qa":
        examples = QADatasetLoader.load(file_path)
    elif task_type == "classification":
        examples = ClassificationDatasetLoader.load(file_path)
    elif task_type == "rag":
        examples = RAGDatasetLoader.load(file_path)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Ensure we have enough examples
    total_needed = train_size + dev_size + test_size
    if len(examples) < total_needed:
        print(
            f"⚠️  Warning: Dataset has {len(examples)} examples, "
            f"but {total_needed} requested. Adjusting sizes..."
        )
        # Scale down proportionally
        scale = len(examples) / total_needed
        train_size = int(train_size * scale)
        dev_size = int(dev_size * scale)
        test_size = len(examples) - train_size - dev_size

    # Use only requested number of examples
    examples = examples[: train_size + dev_size + test_size]

    # Split
    total = train_size + dev_size + test_size
    train_ratio = train_size / total
    dev_ratio = dev_size / total
    test_ratio = test_size / total

    return split_dataset(examples, train_ratio, dev_ratio, test_ratio, random_seed)
