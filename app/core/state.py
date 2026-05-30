"""In-memory application state (embeddings + LLM only; RAG indexes are per-workspace on disk)."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AppState:
    llm: Any = None
    embeddings_model: Any = None

    def is_llm_ready(self) -> bool:
        return self.llm is not None

    def is_api_ready(self) -> bool:
        return self.embeddings_model is not None and self.is_llm_ready()


state = AppState()
