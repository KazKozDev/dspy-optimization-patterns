"""
DSPy Optimization Pipeline - The Compilation Step

This is the CORE of DSPy's value proposition:
Instead of manually writing prompts, you define what success looks like (metric)
and let DSPy compile optimal prompts + few-shot examples.

Key Concepts:
1. Teacher-Student: Use strong model during optimization, deploy with weaker model
2. Compilation != Training: No gradient updates, just prompt engineering
3. Artifacts: Save compiled programs as JSON for deployment

Usage:
    python -m src.pipeline.optimizer \\
        --config config/optimizers.yaml \\
        --module SimpleRAG \\
        --output artifacts/compiled_programs/rag_v1.json
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import dspy
import yaml
from dspy.teleprompt import (
    MIPRO,
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    SignatureOptimizer,
)

from src.core.metrics import debug_metric_failures, get_metric
from src.core.modules import DocumentClassifier, MultiHopReasoner, SimpleRAG
from src.pipeline.loader import load_and_split

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_models(models_config_path: str = "config/models.yaml"):
    """
    Initialize teacher and student models.

    Returns:
        (teacher_lm, student_lm)
    """
    config = load_config(models_config_path)

    # Teacher model (for optimization)
    teacher_cfg = config["teacher"]
    if teacher_cfg["provider"] == "openai":
        teacher_lm = dspy.LM(
            model=f"openai/{teacher_cfg['model']}",
            temperature=teacher_cfg["temperature"],
            max_tokens=teacher_cfg["max_tokens"],
        )
    elif teacher_cfg["provider"] == "anthropic":
        teacher_lm = dspy.LM(
            model=f"anthropic/{teacher_cfg['model']}",
            temperature=teacher_cfg["temperature"],
            max_tokens=teacher_cfg["max_tokens"],
        )
    else:
        raise ValueError(f"Unknown provider: {teacher_cfg['provider']}")

    # Student model (for production)
    student_cfg = config["student"]
    if student_cfg["provider"] == "openai":
        student_lm = dspy.LM(
            model=f"openai/{student_cfg['model']}",
            temperature=student_cfg["temperature"],
            max_tokens=student_cfg["max_tokens"],
        )
    elif student_cfg["provider"] == "ollama":
        student_lm = dspy.LM(
            model=f"ollama/{student_cfg['model']}",
            temperature=student_cfg["temperature"],
            max_tokens=student_cfg["max_tokens"],
        )
    else:
        raise ValueError(f"Unknown provider: {student_cfg['provider']}")

    print(f"✓ Teacher: {teacher_cfg['model']}")
    print(f"✓ Student: {student_cfg['model']}")

    return teacher_lm, student_lm


# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================


def create_optimizer(
    optimizer_type: str,
    metric: Callable,
    teacher_lm: dspy.LM,
    student_lm: dspy.LM,
    config: dict,
):
    """
    Create optimizer based on type.

    Available optimizers:
    - BootstrapFewShot: Generate few-shot examples from training data
    - BootstrapFewShotWithRandomSearch: Try multiple prompt variations
    - MIPRO: Generate optimized instructions + few-shot examples
    - SignatureOptimizer: Refine input/output field descriptions
    """
    if optimizer_type == "BootstrapFewShot":
        return BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=config.get("max_bootstrapped_demos", 8),
            max_labeled_demos=config.get("max_labeled_demos", 4),
            teacher_settings={"lm": teacher_lm},
        )

    elif optimizer_type == "BootstrapFewShotWithRandomSearch":
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=config.get("max_bootstrapped_demos", 8),
            max_labeled_demos=config.get("max_labeled_demos", 4),
            num_candidate_programs=config.get("num_candidate_programs", 10),
            num_threads=config.get("num_threads", 4),
            teacher_settings={"lm": teacher_lm},
        )

    elif optimizer_type == "MIPRO":
        return MIPRO(
            metric=metric,
            num_candidates=config.get("num_candidates", 10),
            init_temperature=config.get("init_temperature", 1.0),
            prompt_model=teacher_lm,
            task_model=student_lm,
        )

    elif optimizer_type == "SignatureOptimizer":
        return SignatureOptimizer(
            metric=metric,
            prompt_model=teacher_lm,
            task_model=student_lm,
            max_iterations=config.get("max_iterations", 5),
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


# ============================================================================
# MAIN OPTIMIZATION WORKFLOW
# ============================================================================


class OptimizationPipeline:
    """
    End-to-end optimization pipeline.

    Steps:
    1. Load and split data
    2. Initialize models (teacher + student)
    3. Create unoptimized module
    4. Run optimizer (compile)
    5. Evaluate on dev set
    6. Save compiled program to artifacts/
    7. Test on hold-out set
    """

    def __init__(
        self,
        models_config: str = "config/models.yaml",
        optimizer_config: str = "config/optimizers.yaml",
    ):
        self.models_config = models_config
        self.optimizer_config = optimizer_config

        # Load configs
        self.opt_config = load_config(optimizer_config)
        self.run_config = self.opt_config["run"]

        # Setup models
        self.teacher_lm, self.student_lm = setup_models(models_config)

    def run(
        self,
        module_class: type,
        data_path: str,
        metric_name: str,
        optimizer_name: str,
        output_path: str,
        task_type: str = "qa",
    ):
        """
        Run complete optimization pipeline.

        Args:
            module_class: DSPy module class (e.g., SimpleRAG)
            data_path: Path to dataset
            metric_name: Name of metric to use (from metrics.py)
            optimizer_name: Type of optimizer (bootstrap, mipro, etc.)
            output_path: Where to save compiled program
            task_type: "qa", "classification", or "rag"
        """
        print("\n" + "=" * 80)
        print(f"OPTIMIZATION PIPELINE: {module_class.__name__}")
        print("=" * 80 + "\n")

        # Step 1: Load and split data
        print("📂 Loading dataset...")
        trainset, devset, testset = load_and_split(
            data_path,
            task_type=task_type,
            train_size=self.run_config["train_size"],
            dev_size=self.run_config["dev_size"],
            test_size=self.run_config["test_size"],
            random_seed=self.run_config["random_seed"],
        )

        # Step 2: Get metric
        print(f"\n🎯 Using metric: {metric_name}")
        metric = get_metric(metric_name)

        # Step 3: Create unoptimized module
        print(f"\n🏗️  Creating module: {module_class.__name__}")
        dspy.settings.configure(lm=self.teacher_lm)

        if module_class == DocumentClassifier:
            # Special case: needs categories
            categories = list({ex.category for ex in trainset})
            module = module_class(categories=categories)
        else:
            module = module_class()

        # Step 4: Create optimizer
        print(f"\n⚙️  Creating optimizer: {optimizer_name}")
        optimizer_config = self.opt_config.get(
            optimizer_name.lower().replace("_", ""),
            {},
        )

        optimizer = create_optimizer(
            optimizer_type=optimizer_config.get("type", optimizer_name),
            metric=metric,
            teacher_lm=self.teacher_lm,
            student_lm=self.student_lm,
            config=optimizer_config,
        )

        # Step 5: COMPILE (this is where the magic happens)
        print("\n🔥 Starting compilation (this may take several minutes)...")
        print(f"   - Train examples: {len(trainset)}")
        print(f"   - Dev examples: {len(devset)}")
        print(f"   - Teacher model: {self.teacher_lm.model}")

        compiled_module = optimizer.compile(
            module,
            trainset=trainset,
            valset=devset,  # DSPy uses this during optimization
        )

        # Step 6: Evaluate on dev set
        print("\n📊 Evaluating compiled module on dev set...")
        from dspy.evaluate import Evaluate

        evaluator = Evaluate(
            devset=devset,
            metric=metric,
            num_threads=self.run_config.get("num_threads", 4),
            display_progress=True,
        )

        dev_score = evaluator(compiled_module)
        print(f"\n✓ Dev Set Score: {dev_score:.2%}")

        # Step 7: Save compiled program
        print("\n💾 Saving compiled program...")
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        compiled_module.save_compiled_state(str(output_path_obj))

        # Step 8: Evaluate on test set (hold-out)
        print("\n🧪 Final evaluation on test set...")
        test_evaluator = Evaluate(
            devset=testset,
            metric=metric,
            num_threads=self.run_config.get("num_threads", 4),
            display_progress=True,
        )

        test_score = test_evaluator(compiled_module)
        print(f"\n✓ Test Set Score: {test_score:.2%}")

        # Step 9: Debug failures (optional)
        failures_path = output_path_obj.parent / f"{output_path_obj.stem}_failures.jsonl"
        debug_metric_failures(
            compiled_module,
            testset[:20],  # Debug first 20 failures
            metric,
            str(failures_path),
        )

        # Summary
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"📈 Dev Score:  {dev_score:.2%}")
        print(f"📈 Test Score: {test_score:.2%}")
        print(f"💾 Saved to:   {output_path}")
        print(f"🐛 Failures:   {failures_path}")
        print("=" * 80 + "\n")

        return compiled_module, dev_score, test_score


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Optimize DSPy modules (compile prompts)")
    parser.add_argument(
        "--module",
        type=str,
        required=True,
        choices=["SimpleRAG", "DocumentClassifier", "MultiHopReasoner"],
        help="Module to optimize",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset (JSONL/JSON/CSV)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="qa",
        choices=["qa", "classification", "rag"],
        help="Type of task",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="hybrid_qa",
        help="Metric name (from metrics.py)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "mipro", "signature_opt"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for compiled program (JSON)",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default="config/models.yaml",
        help="Path to models config",
    )
    parser.add_argument(
        "--optimizer-config",
        type=str,
        default="config/optimizers.yaml",
        help="Path to optimizer config",
    )

    args = parser.parse_args()

    # Map module name to class
    module_map = {
        "SimpleRAG": SimpleRAG,
        "DocumentClassifier": DocumentClassifier,
        "MultiHopReasoner": MultiHopReasoner,
    }

    # Create and run pipeline
    pipeline = OptimizationPipeline(
        models_config=args.models_config,
        optimizer_config=args.optimizer_config,
    )

    pipeline.run(
        module_class=module_map[args.module],
        data_path=args.data,
        metric_name=args.metric,
        optimizer_name=args.optimizer,
        output_path=args.output,
        task_type=args.task_type,
    )


if __name__ == "__main__":
    main()
