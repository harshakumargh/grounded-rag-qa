"""Runtime settings, read from environment variables or a local .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

SAMPLE_SOURCE = (
    "https://raw.githubusercontent.com/tnahddisttud/sample-doc/"
    "refs/heads/main/atliqai_hr_policies.txt"
)


@dataclass(frozen=True)
class Settings:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "openai/gpt-oss-20b"
    # Qdrant: a server URL if given, otherwise embedded local mode at qdrant_path.
    qdrant_url: str | None = None
    qdrant_path: str = "./qdrant_data"
    collection: str = "documents"
    # Measured in cl100k_base tokens. all-MiniLM-L6-v2 truncates input at 256
    # word pieces, so chunks stay well under that.
    chunk_size: int = 128
    chunk_overlap: int = 16
    top_k: int = 5
    # Minimum cosine similarity for a chunk to count as relevant.
    score_threshold: float = 0.3

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        env = os.environ
        return cls(
            embed_model=env.get("EMBED_MODEL", cls.embed_model),
            llm_model=env.get("LLM_MODEL", cls.llm_model),
            qdrant_url=env.get("QDRANT_URL") or None,
            qdrant_path=env.get("QDRANT_PATH", cls.qdrant_path),
            collection=env.get("QDRANT_COLLECTION", cls.collection),
            chunk_size=int(env.get("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(env.get("CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(env.get("TOP_K", cls.top_k)),
            score_threshold=float(env.get("SCORE_THRESHOLD", cls.score_threshold)),
        )
