"""
Vector Database Integration for RAG

Supports multiple vector DB providers:
- Qdrant (open-source, recommended for self-hosting)
- Pinecone (managed service, good for production)
- Weaviate (hybrid search capabilities)
- Chroma (lightweight, good for development)

Each provider implements a common interface for easy switching.
"""

import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class SearchResult:
    """Standardized search result across all providers."""

    text: str
    score: float
    metadata: dict[str, Any]
    id: str


class VectorDBInterface(Protocol):
    """Protocol for vector database implementations."""

    def search(self, query: str, top_k: int = 5, filter: dict | None = None) -> list[SearchResult]:
        """Search for similar documents."""
        ...

    def upsert(self, texts: list[str], metadata: list[dict], ids: list[str] | None = None):
        """Insert or update documents."""
        ...

    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        ...


# ============================================================================
# QDRANT INTEGRATION
# ============================================================================


class QdrantRetriever:
    """
    Qdrant vector database integration.

    Qdrant is open-source and can be self-hosted. Great for:
    - Production deployments with data privacy requirements
    - High-throughput scenarios (handles millions of vectors)
    - Advanced filtering and hybrid search

    Installation:
        pip install qdrant-client sentence-transformers

    Usage:
        retriever = QdrantRetriever(
            host="localhost",
            port=6333,
            collection_name="documents"
        )
        results = retriever.search("What is DSPy?", top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: str | None = None,
    ):
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Qdrant integration requires: pip install qdrant-client sentence-transformers"
            ) from e

        self.collection_name = collection_name

        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port, api_key=api_key)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✓ Created Qdrant collection: {self.collection_name}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        # Embed query
        query_vector = self.embedding_model.encode(query).tolist()

        # Search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter,
        )

        # Convert to standardized format
        results = [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata=hit.payload,
                id=str(hit.id),
            )
            for hit in search_result
        ]

        return results

    def upsert(
        self,
        texts: list[str],
        metadata: list[dict],
        ids: list[str] | None = None,
    ):
        """Insert or update documents."""
        from qdrant_client.models import PointStruct

        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in texts]

        # Embed texts
        vectors = self.embedding_model.encode(texts).tolist()

        # Create points
        points = [
            PointStruct(
                id=id_,
                vector=vector,
                payload={**meta, "text": text},
            )
            for id_, vector, text, meta in zip(ids, vectors, texts, metadata, strict=False)
        ]

        # Upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(f"✓ Upserted {len(texts)} documents to Qdrant")

    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )


# ============================================================================
# PINECONE INTEGRATION
# ============================================================================


class PineconeRetriever:
    """
    Pinecone managed vector database integration.

    Pinecone is a fully-managed service. Great for:
    - Quick prototyping without infrastructure setup
    - Teams without ML Ops expertise
    - Applications with variable load (auto-scaling)

    Installation:
        pip install pinecone-client sentence-transformers

    Usage:
        retriever = PineconeRetriever(
            api_key="your-api-key",
            environment="us-west1-gcp",
            index_name="documents"
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        environment: str = "us-west1-gcp",
        index_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        try:
            import pinecone
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Pinecone integration requires: pip install pinecone-client sentence-transformers"
            ) from e

        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key required (env: PINECONE_API_KEY)")

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        self.index_name = index_name
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Create index if it doesn't exist
        self._ensure_index()
        self.index = pinecone.Index(index_name)

    def _ensure_index(self):
        """Create index if it doesn't exist."""
        import pinecone

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
            )
            print(f"✓ Created Pinecone index: {self.index_name}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        # Embed query
        query_vector = self.embedding_model.encode(query).tolist()

        # Search
        search_result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )

        # Convert to standardized format
        results = [
            SearchResult(
                text=match.metadata.get("text", ""),
                score=match.score,
                metadata=match.metadata,
                id=match.id,
            )
            for match in search_result.matches
        ]

        return results

    def upsert(
        self,
        texts: list[str],
        metadata: list[dict],
        ids: list[str] | None = None,
    ):
        """Insert or update documents."""
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in texts]

        # Embed texts
        vectors = self.embedding_model.encode(texts).tolist()

        # Create upsert data
        to_upsert = [
            (id_, vector, {**meta, "text": text})
            for id_, vector, text, meta in zip(ids, vectors, texts, metadata, strict=False)
        ]

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(to_upsert), batch_size):
            batch = to_upsert[i : i + batch_size]
            self.index.upsert(vectors=batch)

        print(f"✓ Upserted {len(texts)} documents to Pinecone")

    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        self.index.delete(ids=ids)


# ============================================================================
# CHROMA INTEGRATION (Lightweight, good for development)
# ============================================================================


class ChromaRetriever:
    """
    Chroma vector database integration.

    Chroma is lightweight and easy to set up. Great for:
    - Local development and testing
    - Small to medium datasets (< 1M vectors)
    - Quick prototyping

    Installation:
        pip install chromadb

    Usage:
        retriever = ChromaRetriever(collection_name="documents")
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = ".chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Chroma integration requires: pip install chromadb sentence-transformers"
            ) from e

        # Initialize Chroma
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer(embedding_model)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "DSPy document collection"},
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        # Embed query
        query_vector = self.embedding_model.encode(query).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter,
        )

        # Convert to standardized format
        search_results = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append(
                    SearchResult(
                        text=doc,
                        score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        id=results["ids"][0][i],
                    )
                )

        return search_results

    def upsert(
        self,
        texts: list[str],
        metadata: list[dict],
        ids: list[str] | None = None,
    ):
        """Insert or update documents."""
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in texts]

        # Embed texts
        vectors = self.embedding_model.encode(texts).tolist()

        # Upsert
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadata,
        )

        print(f"✓ Upserted {len(texts)} documents to Chroma")

    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        self.collection.delete(ids=ids)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_retriever(
    provider: str = "chroma",
    **kwargs,
) -> VectorDBInterface:
    """
    Factory function to create retriever based on provider.

    Args:
        provider: "qdrant", "pinecone", or "chroma"
        **kwargs: Provider-specific arguments

    Returns:
        Retriever instance

    Usage:
        # For development
        retriever = create_retriever("chroma")

        # For production
        retriever = create_retriever(
            "qdrant",
            host="localhost",
            port=6333,
            collection_name="my_docs"
        )
    """
    providers = {
        "qdrant": QdrantRetriever,
        "pinecone": PineconeRetriever,
        "chroma": ChromaRetriever,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")

    return providers[provider](**kwargs)


# ============================================================================
# UTILITY: BULK INGESTION
# ============================================================================


def ingest_documents_from_file(
    retriever: VectorDBInterface,
    file_path: str,
    batch_size: int = 100,
):
    """
    Ingest documents from a JSONL file into vector DB.

    File format:
    {"text": "Document text...", "metadata": {"source": "..."}}

    Usage:
        retriever = create_retriever("chroma")
        ingest_documents_from_file(retriever, "data/documents.jsonl")
    """
    import json

    texts = []
    metadatas = []

    with open(file_path) as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                texts.append(doc["text"])
                metadatas.append(doc.get("metadata", {}))

                # Upsert in batches
                if len(texts) >= batch_size:
                    retriever.upsert(texts, metadatas)
                    texts = []
                    metadatas = []

    # Upsert remaining
    if texts:
        retriever.upsert(texts, metadatas)

    print(f"✓ Ingestion complete from {file_path}")
