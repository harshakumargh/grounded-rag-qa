"""Vector store: embed chunks and index them in Qdrant, or reopen an index."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from grounded_rag.config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    return HuggingFaceEmbeddings(model_name=settings.embed_model)


def _connection(settings: Settings) -> dict:
    if settings.qdrant_url:
        return {"url": settings.qdrant_url}
    if settings.qdrant_path == ":memory:":
        return {"location": ":memory:"}
    return {"path": settings.qdrant_path}


def index_documents(
    chunks: list[Document], embeddings: Embeddings, settings: Settings
) -> QdrantVectorStore:
    """Embed chunks and (re)build the collection. force_recreate makes
    re-ingesting idempotent instead of appending duplicate points."""
    return QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        collection_name=settings.collection,
        force_recreate=True,
        **_connection(settings),
    )


def open_store(embeddings: Embeddings, settings: Settings) -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=settings.collection,
        **_connection(settings),
    )
