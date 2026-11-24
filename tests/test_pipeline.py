"""Offline tests: fake embeddings, in-memory Qdrant and a fake chat model,
so no network access or API key is needed."""

import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_qdrant import QdrantVectorStore

from grounded_rag.chain import FALLBACK, build_chain, format_docs
from grounded_rag.ingest import ENCODING, split_documents


def test_chunks_stay_within_token_budget():
    long_paragraph = " ".join(f"word{i}" for i in range(2000))
    docs = [Document(page_content=long_paragraph, metadata={"source": "test"})]

    chunks = split_documents(docs, chunk_size=64, chunk_overlap=8)

    enc = tiktoken.get_encoding(ENCODING)
    assert len(chunks) > 1
    assert all(len(enc.encode(c.page_content)) <= 64 for c in chunks)
    assert all(c.metadata["source"] == "test" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_format_docs_numbers_passages():
    docs = [Document(page_content="alpha"), Document(page_content="beta")]
    assert format_docs(docs) == "[1]\nalpha\n\n---\n\n[2]\nbeta"


def test_score_threshold_filters_irrelevant_chunks():
    texts = ["casual leave policy", "notice period policy", "probation policy"]
    store = QdrantVectorStore.from_texts(
        texts,
        embedding=DeterministicFakeEmbedding(size=64),
        location=":memory:",
        collection_name="test",
    )

    # Plain top-k always returns k results, relevant or not.
    plain = store.as_retriever(search_kwargs={"k": 3})
    assert len(plain.invoke("completely unrelated question")) == 3

    # With a similarity floor, only the exact match (cosine 1.0) survives.
    strict = store.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.99})
    hits = strict.invoke("notice period policy")
    assert [d.page_content for d in hits] == ["notice period policy"]
    assert strict.invoke("completely unrelated question") == []


def test_chain_returns_fallback_without_calling_llm():
    empty_retriever = RunnableLambda(lambda q: [])
    llm = FakeListChatModel(responses=[])  # raises if it is ever called

    result = build_chain(empty_retriever, llm).invoke("anything")

    assert result == {"answer": FALLBACK, "sources": []}


def test_chain_answers_from_retrieved_docs():
    docs = [Document(page_content="Employees get 12 casual leaves per year.")]
    retriever = RunnableLambda(lambda q: docs)
    llm = FakeListChatModel(responses=["12 casual leaves per year [1]."])

    result = build_chain(retriever, llm).invoke("How many casual leaves?")

    assert result["answer"] == "12 casual leaves per year [1]."
    assert result["sources"] == docs
