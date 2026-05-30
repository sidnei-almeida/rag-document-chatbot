"""Centralized environment configuration for DocMind."""

import os
from dataclasses import dataclass


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = _env_float("GROQ_TEMPERATURE", "0.15")
    GROQ_MAX_TOKENS: int = _env_int("GROQ_MAX_TOKENS", "1024")

    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    WORKSPACE_STORAGE_ROOT: str = os.getenv(
        "WORKSPACE_STORAGE_ROOT",
        os.getenv("WORKSPACE_STORAGE_PATH", "/tmp/docmind_storage/workspaces"),
    )

    CHUNK_SIZE: int = _env_int("CHUNK_SIZE", "1200")
    CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", "180")
    RETRIEVAL_K: int = _env_int("RETRIEVAL_K", "6")
    RETRIEVAL_FETCH_K: int = _env_int("RETRIEVAL_FETCH_K", "20")
    RETRIEVAL_LAMBDA: float = _env_float("RETRIEVAL_LAMBDA", "0.7")
    PREVIEW_MAX_LENGTH: int = _env_int("PREVIEW_MAX_LENGTH", "320")

    MAX_FILE_SIZE_MB: int = _env_int("MAX_FILE_SIZE_MB", "20")
    MAX_FILES_PER_WORKSPACE: int = _env_int("MAX_FILES_PER_WORKSPACE", "5")
    MAX_PAGES_PER_FILE: int = _env_int("MAX_PAGES_PER_FILE", "40")
    MAX_TOTAL_PAGES: int = _env_int("MAX_TOTAL_PAGES", "100")
    MAX_QUESTION_LENGTH: int = _env_int("MAX_QUESTION_LENGTH", "1000")

    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    SAMPLE_DOCUMENTS_DIR: str = os.getenv("SAMPLE_DOCUMENTS_DIR", "sample_documents")
    SAMPLE_DOCUMENT_FILENAME: str = os.getenv(
        "SAMPLE_DOCUMENT_FILENAME", "ai-document-intelligence-report.pdf"
    )
    AUTO_LOAD_SAMPLE_ON_STARTUP: bool = os.getenv(
        "AUTO_LOAD_SAMPLE_ON_STARTUP", "false"
    ).lower() in ("1", "true", "yes")

    TEXT_SPLITTER_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ".", " ", "")
    NO_EVIDENCE_MESSAGE: str = (
        "I could not find enough evidence in the uploaded documents to answer that reliably."
    )
    GENERAL_QUESTIONS: tuple[str, ...] = (
        "hello",
        "hi",
        "hey",
        "how are you",
        "what can you do",
        "help",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "olá",
        "oi",
    )

    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def limits_dict(self) -> dict[str, int]:
        return {
            "max_file_size_mb": self.MAX_FILE_SIZE_MB,
            "max_files_per_workspace": self.MAX_FILES_PER_WORKSPACE,
            "max_pages_per_file": self.MAX_PAGES_PER_FILE,
            "max_total_pages": self.MAX_TOTAL_PAGES,
            "max_question_length": self.MAX_QUESTION_LENGTH,
        }

    def retrieval_dict(self) -> dict[str, int | str | float]:
        return {
            "type": "mmr",
            "k": self.RETRIEVAL_K,
            "fetch_k": self.RETRIEVAL_FETCH_K,
            "lambda": self.RETRIEVAL_LAMBDA,
        }


settings = Settings()

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
)


def get_cors_origins() -> list[str]:
    """CORS origins: env list, or defaults for local frontends + wildcard if explicitly set."""
    raw = settings.ALLOWED_ORIGINS.strip()
    if raw == "*":
        return ["*"]
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return list(DEFAULT_CORS_ORIGINS)
