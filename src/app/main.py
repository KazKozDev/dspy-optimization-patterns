"""
FastAPI Application for DSPy Inference

This is the production serving layer.
It loads compiled DSPy programs from artifacts/ and serves predictions via REST API.

Key Concepts:
1. Load compiled program on startup (not on every request!)
2. Use student model (cheap, fast) for inference
3. Log all requests for monitoring
4. Handle errors gracefully
5. Provide health checks for orchestration

Usage:
    uvicorn src.app.main:app --reload --port 8000

Docker:
    docker build -t dspy-api .
    docker run -p 8000:8000 -e OPENAI_API_KEY=xxx dspy-api
"""

import os
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import dspy

from src.core.modules import SimpleRAG, DocumentClassifier, load_module_from_artifact
from src.app.schemas import (
    QuestionRequest,
    QuestionResponse,
    ClassificationRequest,
    ClassificationResponse,
    RAGRequest,
    RAGResponse,
    HealthResponse,
    ErrorResponse,
)
from src.utils.tracing import DSPyTracer, setup_logging


# ============================================================================
# GLOBAL STATE
# ============================================================================


class AppState:
    """Container for application state."""

    def __init__(self):
        self.rag_module: Optional[SimpleRAG] = None
        self.classifier_module: Optional[DocumentClassifier] = None
        self.model_version: str = "unoptimized"
        self.tracer: DSPyTracer = DSPyTracer()
        self.start_time: float = time.time()
        self.lm: Optional[dspy.LM] = None


state = AppState()


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.

    Startup:
    - Load environment variables
    - Initialize student LM
    - Load compiled programs from artifacts/
    - Setup tracing

    Shutdown:
    - Cleanup resources
    """
    # STARTUP
    print("\n" + "=" * 80)
    print("🚀 Starting DSPy Production API")
    print("=" * 80 + "\n")

    # 1. Setup logging
    logger = setup_logging()
    logger.info("Logger initialized")

    # 2. Load API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set. API calls will fail.")

    # 3. Initialize student model (production model)
    student_model = os.getenv("STUDENT_MODEL", "gpt-5-mini")

    # GPT-5 models require temperature=1.0 and max_tokens >= 16000
    if "gpt-5" in student_model:
        state.lm = dspy.LM(
            model=f"openai/{student_model}",
            temperature=1.0,
            max_tokens=16000,
        )
    else:
        state.lm = dspy.LM(
            model=f"openai/{student_model}",
            temperature=0.0,
            max_tokens=1500,
        )
    dspy.settings.configure(lm=state.lm)
    print(f"✓ Configured student model: {student_model}")

    # 4. Load compiled programs
    artifacts_dir = Path("artifacts/compiled_programs")

    # Try to load RAG module
    rag_artifacts = list(artifacts_dir.glob("rag_*.json"))
    if rag_artifacts:
        latest_rag = sorted(rag_artifacts)[-1]  # Get latest version
        try:
            state.rag_module = SimpleRAG()
            state.rag_module.load_compiled_state(str(latest_rag))
            state.model_version = latest_rag.stem
            print(f"✓ Loaded RAG module: {latest_rag.name}")
        except Exception as e:
            print(f"⚠️  Failed to load RAG module: {e}")
            state.rag_module = SimpleRAG()  # Use unoptimized

    # Try to load classifier
    classifier_artifacts = list(artifacts_dir.glob("classifier_*.json"))
    if classifier_artifacts:
        latest_classifier = sorted(classifier_artifacts)[-1]
        try:
            state.classifier_module = load_module_from_artifact(
                str(latest_classifier), DocumentClassifier
            )
            print(f"✓ Loaded classifier: {latest_classifier.name}")
        except Exception as e:
            print(f"⚠️  Failed to load classifier: {e}")
            # Fallback to unoptimized classifier
            state.classifier_module = DocumentClassifier(
                categories=["research", "news", "tutorial", "opinion", "other"]
            )
    else:
        # No compiled classifier found, use unoptimized version with default categories
        state.classifier_module = DocumentClassifier(
            categories=["research", "news", "tutorial", "opinion", "other"]
        )

    # If no compiled programs found, use unoptimized modules
    if not state.rag_module:
        print("⚠️  No compiled RAG module found. Using unoptimized version.")
        state.rag_module = SimpleRAG()

    print("\n✓ API ready to serve requests")
    print("=" * 80 + "\n")

    yield  # Server runs here

    # SHUTDOWN
    print("\n🛑 Shutting down DSPy API")


# ============================================================================
# FASTAPI APP
# ============================================================================


app = FastAPI(
    title="DSPy Production API",
    description="REST API for serving optimized DSPy modules",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HEALTH CHECK
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.rag_module is not None,
        model_version=state.model_version,
        uptime_seconds=time.time() - state.start_time,
    )


# ============================================================================
# QUESTION ANSWERING ENDPOINTS
# ============================================================================


@app.post("/qa", response_model=QuestionResponse)
async def question_answering(request: QuestionRequest):
    """
    Answer a question using the loaded DSPy module.

    If context is provided, uses it directly.
    Otherwise, would integrate with a retriever (not implemented here).
    """
    try:
        start_time = time.time()
        logs = []

        if not state.rag_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG module not loaded",
            )

        logs.append("📝 Initializing question answering module...")

        # For this endpoint, we use provided context or empty
        # In production, integrate with a vector DB here
        if request.context:
            logs.append("🔍 Processing question with provided context...")
            # Simple QA with provided context
            prediction = state.rag_module.generate_answer(
                context=request.context,
                question=request.question,
            )
            logs.append("✓ Generated answer using DSPy optimized prompts")
            answer = prediction.answer
            reasoning = getattr(prediction, "reasoning", None)
            prompt_used = getattr(prediction, "prompt_used", None)
        else:
            # Would call retriever here
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Context is required for this endpoint. Use /rag for retrieval.",
            )

        # Log request
        latency_ms = (time.time() - start_time) * 1000
        logs.append(f"⏱️ Completed in {latency_ms:.1f}ms")

        state.tracer.log_prediction(
            module_name="SimpleRAG",
            inputs=request.model_dump(),
            prediction=prediction,
            latency_ms=latency_ms,
        )

        return QuestionResponse(
            answer=answer,
            reasoning=reasoning,
            model_version=state.model_version,
            prompt_used=prompt_used,
            execution_time_ms=latency_ms,
            logs=logs,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest):
    """
    RAG endpoint with retrieval.

    Note: This is a mock implementation.
    In production, integrate with:
    - Vector database (Pinecone, Weaviate, Qdrant)
    - Embedding model (OpenAI, Cohere, local)
    - Document store
    """
    try:
        start_time = time.time()
        logs = []

        if not state.rag_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG module not loaded",
            )

        logs.append("🔍 Starting RAG pipeline...")

        # Mock retriever function
        def mock_retriever(query: str) -> List[str]:
            """
            Placeholder retriever.

            In production, replace with:
                def real_retriever(query: str) -> List[str]:
                    embeddings = embed_model.embed(query)
                    results = vector_db.search(embeddings, top_k=request.top_k)
                    return [doc.text for doc in results]
            """
            logs.append(f"📚 Retrieving top {request.top_k} documents...")
            return [
                f"Mock context 1 for query: {query}",
                f"Mock context 2 for query: {query}",
                f"Mock context 3 for query: {query}",
            ]

        retrieval_start = time.time()
        # Run RAG pipeline
        prediction = state.rag_module.forward(
            question=request.question,
            retriever_fn=mock_retriever,
            conversation_history=request.conversation_history,
        )
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        logs.append("✓ Retrieved documents and generated answer")

        # Log request
        latency_ms = (time.time() - start_time) * 1000
        logs.append(f"⏱️ Completed in {latency_ms:.1f}ms (retrieval: {retrieval_time_ms:.1f}ms)")

        state.tracer.log_prediction(
            module_name="SimpleRAG",
            inputs=request.model_dump(),
            prediction=prediction,
            latency_ms=latency_ms,
            metadata={"endpoint": "rag"},
        )

        prompt_used = getattr(prediction, "prompt_used", None)

        return RAGResponse(
            answer=prediction.answer,
            sources=prediction.contexts,
            search_query=prediction.search_query,
            reasoning=getattr(prediction, "reasoning", None),
            model_version=state.model_version,
            prompt_used=prompt_used,
            execution_time_ms=latency_ms,
            retrieval_time_ms=retrieval_time_ms,
            logs=logs,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ============================================================================
# CLASSIFICATION ENDPOINT
# ============================================================================


@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(request: ClassificationRequest):
    """
    Classify document into categories.

    Requires classifier module to be loaded.
    """
    try:
        start_time = time.time()
        logs = []

        if not state.classifier_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Classifier module not loaded",
            )

        logs.append("📁 Initializing classification module...")
        logs.append(f"📖 Processing text ({len(request.text)} chars)...")

        # Run classification
        prediction = state.classifier_module(document_text=request.text)

        logs.append("✓ Classification complete using DSPy optimized prompts")

        # Log request
        latency_ms = (time.time() - start_time) * 1000
        logs.append(f"⏱️ Completed in {latency_ms:.1f}ms")

        state.tracer.log_prediction(
            module_name="DocumentClassifier",
            inputs=request.model_dump(),
            prediction=prediction,
            latency_ms=latency_ms,
        )

        prompt_used = getattr(prediction, "prompt_used", None)
        confidence = getattr(prediction, "confidence", None)

        return ClassificationResponse(
            category=prediction.category,
            reasoning=getattr(prediction, "reasoning", None),
            model_version=state.classifier_module.version,
            confidence=confidence,
            prompt_used=prompt_used,
            execution_time_ms=latency_ms,
            logs=logs,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "DSPy Production API",
        "version": "1.0.0",
        "status": "running",
        "model_version": state.model_version,
        "endpoints": {
            "health": "/health",
            "qa": "/qa (POST)",
            "rag": "/rag (POST)",
            "classify": "/classify (POST)",
            "artifacts": "/artifacts (GET)",
            "docs": "/docs",
        },
    }


# ============================================================================
# ARTIFACTS MANAGEMENT ENDPOINTS
# ============================================================================


@app.get("/artifacts")
async def list_artifacts():
    """
    List all available compiled programs (artifacts).

    Returns information about all compiled DSPy programs in artifacts/ directory.
    """
    try:
        artifacts_dir = Path("artifacts/compiled_programs")
        artifacts = []

        if artifacts_dir.exists():
            for artifact_file in artifacts_dir.glob("*.json"):
                # Get file stats
                stat = artifact_file.stat()

                # Determine type from filename
                artifact_type = "Unknown"
                if "rag" in artifact_file.name.lower():
                    artifact_type = "RAG"
                elif "classifier" in artifact_file.name.lower():
                    artifact_type = "Classifier"
                elif "qa" in artifact_file.name.lower():
                    artifact_type = "QA"

                # Check if this is the currently loaded artifact
                is_active = (state.model_version == artifact_file.stem)

                artifacts.append({
                    "name": artifact_file.name,
                    "type": artifact_type,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "active": is_active,
                    "path": str(artifact_file)
                })

        # Sort by modified time (newest first)
        artifacts.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "total": len(artifacts),
            "artifacts": artifacts
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing artifacts: {str(e)}"
        )


@app.post("/artifacts/{artifact_name}/activate")
async def activate_artifact(artifact_name: str):
    """
    Activate a specific artifact (switch to using a different compiled program).

    This allows hot-swapping between different optimized versions without restarting.
    """
    try:
        artifact_path = Path(f"artifacts/compiled_programs/{artifact_name}")

        if not artifact_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact not found: {artifact_name}"
            )

        # Determine module type and reload
        if "rag" in artifact_name.lower():
            state.rag_module = SimpleRAG()
            state.rag_module.load_compiled_state(str(artifact_path))
            state.model_version = artifact_path.stem

            return {
                "success": True,
                "message": f"Successfully activated RAG artifact: {artifact_name}",
                "active_version": state.model_version
            }
        elif "classifier" in artifact_name.lower():
            state.classifier_module = DocumentClassifier(
                categories=["research", "news", "tutorial", "opinion", "other"]
            )
            state.classifier_module.load_compiled_state(str(artifact_path))

            return {
                "success": True,
                "message": f"Successfully activated Classifier artifact: {artifact_name}",
                "active_version": artifact_path.stem
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not determine artifact type from filename"
            )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activating artifact: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """
    Get API statistics and metrics.

    Returns information about request counts, performance, and system status.
    """
    uptime_seconds = time.time() - state.start_time

    return {
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m",
        "active_model": state.model_version,
        "modules_loaded": {
            "rag": state.rag_module is not None,
            "classifier": state.classifier_module is not None
        },
        "api_status": "healthy" if state.rag_module is not None else "degraded",
        "timestamp": time.time()
    }


# ============================================================================
# MAIN (for local development)
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
