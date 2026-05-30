"""In-memory application state for vector store, LLM, and index metadata."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AppState:
    vector_store: Any = None
    retriever: Any = None
    llm: Any = None
    embeddings_model: Any = None
    last_indexed_filename: str = "document.pdf"
    index_pages: int = 0
    index_chunks: int = 0
    index_cleared: bool = False
    retriever_type: str = "mmr"

    def is_index_ready(self) -> bool:
        return (
            not self.index_cleared
            and self.retriever is not None
            and self.vector_store is not None
        )

    def document_loaded(self) -> bool:
        return self.is_index_ready()

    def reset_index_metadata(self) -> None:
        self.index_pages = 0
        self.index_chunks = 0
        self.last_indexed_filename = "document.pdf"


state = AppState()
