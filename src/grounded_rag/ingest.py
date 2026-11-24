"""Ingestion: load a source document and split it into token-bounded chunks."""

from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ENCODING = "cl100k_base"


def load_source(source: str) -> list[Document]:
    """Load a local text/Markdown file or an http(s) URL as a single Document."""
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        text = response.text
    else:
        text = Path(source).read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": source})]


def split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split on paragraph, then line, then word boundaries, with chunk length
    counted in tokens. Unlike CharacterTextSplitter, an oversized paragraph is
    split further instead of being kept whole, so no chunk exceeds chunk_size.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
