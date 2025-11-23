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
    student_model = os.getenv("STUDENT_MODEL", "gpt-4o-mini")
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

        if not state.rag_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG module not loaded",
            )

        # For this endpoint, we use provided context or empty
        # In production, integrate with a vector DB here
        if request.context:
            # Simple QA with provided context
            prediction = state.rag_module.generate_answer(
                context=request.context,
                question=request.question,
            )
            answer = prediction.answer
            reasoning = getattr(prediction, "reasoning", None)
        else:
            # Would call retriever here
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Context is required for this endpoint. Use /rag for retrieval.",
            )

        # Log request
        latency_ms = (time.time() - start_time) * 1000
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

        if not state.rag_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG module not loaded",
            )

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
            return [
                f"Mock context 1 for query: {query}",
                f"Mock context 2 for query: {query}",
                f"Mock context 3 for query: {query}",
            ]

        # Run RAG pipeline
        prediction = state.rag_module.forward(
            question=request.question,
            retriever_fn=mock_retriever,
            conversation_history=request.conversation_history,
        )

        # Log request
        latency_ms = (time.time() - start_time) * 1000
        state.tracer.log_prediction(
            module_name="SimpleRAG",
            inputs=request.model_dump(),
            prediction=prediction,
            latency_ms=latency_ms,
            metadata={"endpoint": "rag"},
        )

        return RAGResponse(
            answer=prediction.answer,
            sources=prediction.contexts,
            search_query=prediction.search_query,
            reasoning=getattr(prediction, "reasoning", None),
            model_version=state.model_version,
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

        if not state.classifier_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Classifier module not loaded",
            )

        # Run classification
        prediction = state.classifier_module(document_text=request.text)

        # Log request
        latency_ms = (time.time() - start_time) * 1000
        state.tracer.log_prediction(
            module_name="DocumentClassifier",
            inputs=request.model_dump(),
            prediction=prediction,
            latency_ms=latency_ms,
        )

        return ClassificationResponse(
            category=prediction.category,
            reasoning=getattr(prediction, "reasoning", None),
            model_version=state.classifier_module.version,
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
            "docs": "/docs",
        },
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
