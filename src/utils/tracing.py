"""
Observability and Tracing for DSPy

In production, you MUST be able to see:
1. What prompts were actually sent to the LLM
2. What few-shot examples were used
3. What the model responded
4. How much each call cost
5. Where failures occurred in multi-step reasoning

Tools:
- Arize Phoenix: Open-source observability (recommended)
- LangSmith: Commercial option
- Custom logging: Simple but limited

This module provides integration with Phoenix and basic logging.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

import dspy


# ============================================================================
# BASIC LOGGING SETUP
# ============================================================================


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
) -> logging.Logger:
    """
    Setup structured logging for DSPy.

    Logs both to file and console with different formats.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("dspy_production")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # File handler (JSON format for parsing)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"dspy_{timestamp}.jsonl"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


# ============================================================================
# ARIZE PHOENIX INTEGRATION
# ============================================================================


def setup_phoenix_tracing(
    project_name: str = "dspy_production",
    endpoint: Optional[str] = None,
) -> None:
    """
    Setup Arize Phoenix for observability.

    Phoenix provides:
    - Trace visualization (see prompt flow)
    - Token usage tracking
    - Latency monitoring
    - Error tracking

    Installation:
        pip install arize-phoenix openinference-instrumentation-dspy

    Usage:
        setup_phoenix_tracing()
        # Now all DSPy calls are automatically traced

    Args:
        project_name: Name for this project in Phoenix
        endpoint: Phoenix server endpoint (default: http://localhost:6006)
    """
    try:
        import phoenix as px
        from openinference.instrumentation.dspy import DSPyInstrumentor

        # Start Phoenix server (if not already running)
        if endpoint is None:
            session = px.launch_app()
            endpoint = session.url
            print(f"🔍 Phoenix UI: {endpoint}")

        # Instrument DSPy
        DSPyInstrumentor().instrument(
            tracer_provider=px.tracer_provider(project_name=project_name)
        )

        print(f"✓ Phoenix tracing enabled for project: {project_name}")
        print(f"   View traces at: {endpoint}")

    except ImportError:
        print(
            "⚠️  Phoenix not installed. Install with:\n"
            "   pip install arize-phoenix openinference-instrumentation-dspy"
        )


# ============================================================================
# CUSTOM TRACE LOGGING
# ============================================================================


class DSPyTracer:
    """
    Custom tracer for DSPy calls.

    Logs:
    - Input to module
    - Output from module
    - Intermediate steps (for ChainOfThought, ReAct)
    - Token usage
    - Latency
    """

    def __init__(self, log_path: str = "logs/traces"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(log_dir=str(self.log_path))

    def log_prediction(
        self,
        module_name: str,
        inputs: Dict[str, Any],
        prediction: dspy.Prediction,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a single prediction."""
        trace_data = {
            "module": module_name,
            "inputs": inputs,
            "outputs": prediction.toDict(),
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if metadata:
            trace_data["metadata"] = metadata

        self.logger.info(
            f"Prediction logged",
            extra={"extra": trace_data},
        )

    @contextmanager
    def trace_module(self, module_name: str, inputs: Dict[str, Any]):
        """
        Context manager for tracing module execution.

        Usage:
            tracer = DSPyTracer()
            with tracer.trace_module("SimpleRAG", {"question": "..."}):
                result = rag_module(question="...")
        """
        import time

        start_time = time.time()

        try:
            yield
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"Module {module_name} completed in {latency_ms:.2f}ms",
                extra={
                    "extra": {
                        "module": module_name,
                        "inputs": inputs,
                        "latency_ms": latency_ms,
                    }
                },
            )


# ============================================================================
# COST TRACKING
# ============================================================================


class CostTracker:
    """
    Track LLM API costs.

    Useful for:
    - Budget monitoring
    - Comparing optimization strategies
    - Detecting expensive queries
    """

    # Pricing per 1M tokens (as of 2025)
    PRICING = {
        "gpt-5o": {"input": 2.0, "output": 8.0},
        "gpt-5o-mini": {"input": 0.10, "output": 0.5},
        "claude-sonnet-4.5": {"input": 2.5, "output": 12.0},
        "claude-haiku-3.5": {"input": 0.8, "output": 4.0},
    }

    def __init__(self):
        self.total_cost = 0.0
        self.call_history = []

    def log_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Log an LLM call and calculate cost.

        Returns:
            Cost in USD
        """
        # Normalize model name
        model_key = model.split("/")[-1]  # Remove provider prefix

        if model_key not in self.PRICING:
            print(f"⚠️  Unknown model for pricing: {model_key}")
            return 0.0

        pricing = self.PRICING[model_key]

        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        # Log
        self.call_history.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": total_cost,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        self.total_cost += total_cost

        return total_cost

    def report(self) -> Dict[str, Any]:
        """Generate cost report."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_calls": len(self.call_history),
            "total_input_tokens": sum(c["input_tokens"] for c in self.call_history),
            "total_output_tokens": sum(c["output_tokens"] for c in self.call_history),
            "avg_cost_per_call": (
                round(self.total_cost / len(self.call_history), 4)
                if self.call_history
                else 0
            ),
        }

    def save_report(self, output_path: str):
        """Save detailed cost report."""
        report = self.report()
        report["detailed_calls"] = self.call_history

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"💰 Cost Report:")
        print(f"   Total Cost: ${report['total_cost_usd']:.4f}")
        print(f"   Total Calls: {report['total_calls']}")
        print(f"   Avg Cost/Call: ${report['avg_cost_per_call']:.4f}")


# ============================================================================
# CACHING FOR OPTIMIZATION
# ============================================================================


def setup_dspy_cache(cache_dir: str = ".cache/dspy"):
    """
    Enable caching for DSPy LLM calls.

    During optimization, you'll call the same prompts multiple times.
    Caching saves money and time.

    DSPy has built-in caching support.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Enable DSPy caching
    dspy.settings.configure(
        experimental=True,
        cache_dir=str(cache_path),
    )

    print(f"✓ DSPy cache enabled: {cache_path}")


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================


def inspect_compiled_program(compiled_program_path: str):
    """
    Inspect a compiled DSPy program.

    Shows:
    - Few-shot examples
    - Optimized instructions
    - Signatures
    """
    with open(compiled_program_path, "r") as f:
        program_data = json.load(f)

    print("\n" + "=" * 80)
    print(f"COMPILED PROGRAM: {compiled_program_path}")
    print("=" * 80 + "\n")

    # Show predictors
    if "predictors" in program_data:
        for predictor_name, predictor_data in program_data["predictors"].items():
            print(f"📌 {predictor_name}")

            if "demos" in predictor_data:
                print(f"   Few-shot examples: {len(predictor_data['demos'])}")
                for i, demo in enumerate(predictor_data["demos"][:2]):  # Show first 2
                    print(f"\n   Example {i + 1}:")
                    print(f"   {demo}")

            if "signature_instructions" in predictor_data:
                print(f"\n   Instructions:")
                print(f"   {predictor_data['signature_instructions']}")

            print()


def debug_prediction(
    module: dspy.Module,
    inputs: Dict[str, Any],
) -> dspy.Prediction:
    """
    Run prediction with detailed debugging output.

    Shows:
    - Actual prompt sent to LLM
    - Model response
    - Parsed output
    """
    print("\n" + "=" * 80)
    print("DEBUG MODE")
    print("=" * 80 + "\n")

    print("📥 Inputs:")
    for key, value in inputs.items():
        print(f"   {key}: {value}")

    # Run prediction
    with dspy.context(show_guidelines=True):
        prediction = module(**inputs)

    print("\n📤 Outputs:")
    for key, value in prediction.toDict().items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 80 + "\n")

    return prediction
