"""Command-line entry point: ingest, ask, search."""

import argparse
import sys

from grounded_rag.config import SAMPLE_SOURCE, Settings


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    from grounded_rag.ingest import load_source, split_documents
    from grounded_rag.store import build_embeddings, index_documents

    docs = load_source(args.source)
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    index_documents(chunks, build_embeddings(settings), settings)
    print(f"Indexed {len(chunks)} chunks from {args.source} into '{settings.collection}'.")


def cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    from grounded_rag.chain import build_chain, build_llm, build_retriever
    from grounded_rag.store import build_embeddings, open_store

    store = open_store(build_embeddings(settings), settings)
    chain = build_chain(build_retriever(store, settings), build_llm(settings))
    result = chain.invoke(args.question)

    print(result["answer"])
    if args.show_context:
        for i, doc in enumerate(result["sources"], 1):
            print(f"\n[{i}] chunk {doc.metadata.get('chunk_id')}\n{doc.page_content}")


def cmd_search(args: argparse.Namespace, settings: Settings) -> None:
    """Show raw similarity scores, useful for picking SCORE_THRESHOLD."""
    from grounded_rag.store import build_embeddings, open_store

    store = open_store(build_embeddings(settings), settings)
    hits = store.similarity_search_with_score(args.question, k=args.k)
    for doc, score in hits:
        mark = "+" if score >= settings.score_threshold else "-"
        preview = " ".join(doc.page_content.split())[:90]
        print(f"{mark} {score:.3f}  {preview}")
    print(f"\n(+ passes the {settings.score_threshold} threshold, - is filtered out)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="grounded-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="load, chunk, embed and index a document")
    ingest.add_argument("--source", default=SAMPLE_SOURCE, help="file path or URL")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="answer a question from the indexed documents")
    ask.add_argument("question")
    ask.add_argument("--show-context", action="store_true", help="print retrieved chunks")
    ask.set_defaults(func=cmd_ask)

    search = sub.add_parser("search", help="show retrieval scores without calling the LLM")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=5)
    search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args, Settings.from_env())


if __name__ == "__main__":
    main()
