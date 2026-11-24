"""Retrieval + grounded generation, composed from LangChain Runnables."""

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStore
from langchain_groq import ChatGroq

from grounded_rag.config import Settings

FALLBACK = "I don't know. The indexed documents don't cover this question."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Do not use outside knowledge. Cite the passages you used, like [1] or [2]. "
    f'If the context does not contain the answer, reply exactly: "{FALLBACK}"'
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def build_llm(settings: Settings) -> BaseChatModel:
    return ChatGroq(
        model=settings.llm_model,
        temperature=0,
        reasoning_format="parsed",
        max_retries=2,
    )


def build_retriever(store: VectorStore, settings: Settings) -> BaseRetriever:
    """Top-k similarity search with a relevance floor. Without the threshold,
    top-k always returns k chunks, even for questions the corpus can't answer."""
    return store.as_retriever(
        search_kwargs={"k": settings.top_k, "score_threshold": settings.score_threshold}
    )


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1)
    )


def build_chain(retriever: Runnable, llm: BaseChatModel) -> Runnable:
    """question -> {"answer": str, "sources": list[Document]}.

    When retrieval returns nothing above the threshold, the chain returns the
    fallback without calling the LLM.
    """
    generate = PROMPT | llm | StrOutputParser()

    def answer(inputs: dict) -> dict:
        docs = inputs["docs"]
        if not docs:
            return {"answer": FALLBACK, "sources": []}
        text = generate.invoke(
            {"context": format_docs(docs), "question": inputs["question"]}
        )
        return {"answer": text, "sources": docs}

    return RunnableParallel(
        docs=retriever, question=RunnablePassthrough()
    ) | RunnableLambda(answer)
