"""Unit tests for RAG retrieval helpers (no external services)."""

from langchain_core.documents import Document

from app.core.config import settings
from app.core.state import state
from app.services.retrieval import (
    build_ask_response,
    clean_preview,
    compute_confidence,
    extract_page_index,
    format_sources,
)


def test_extract_page_index_defaults_to_zero():
    assert extract_page_index({}) == 0
    assert extract_page_index({"page": 3}) == 3


def test_page_is_one_indexed_in_sources():
    state.last_indexed_filename = "report.pdf"
    docs = [
        Document(
            page_content="Revenue grew in Q4.",
            metadata={"page": 0, "file_name": "report.pdf", "chunk_id": 7},
        )
    ]
    sources = format_sources(docs, scores=[0.25])
    assert len(sources) == 1
    assert sources[0]["page_index"] == 0
    assert sources[0]["page"] == 1
    assert sources[0]["file_name"] == "report.pdf"
    assert sources[0]["chunk_id"] == 7
    assert sources[0]["score"] == 0.25


def test_clean_preview_collapses_whitespace_and_truncates():
    long_text = "word " * 200
    preview = clean_preview(f"  line1\n\n  line2   {long_text}")
    assert "\n" not in preview
    assert "  " not in preview
    assert len(preview) <= settings.PREVIEW_MAX_LENGTH


def test_format_sources_skips_empty_chunks():
    docs = [
        Document(page_content="   ", metadata={"page": 0}),
        Document(page_content="Valid content.", metadata={"page": 1}),
    ]
    sources = format_sources(docs)
    assert len(sources) == 1
    assert sources[0]["page"] == 2
    assert sources[0]["page_index"] == 1


def test_compute_confidence_levels():
    label, reason = compute_confidence(0, has_context=False)
    assert label == "low"
    assert "No reliable" in reason

    label, _ = compute_confidence(1, has_context=True)
    assert label == "low"

    label, _ = compute_confidence(2, has_context=True)
    assert label == "medium"

    label, reason = compute_confidence(5, has_context=True)
    assert label == "high"
    assert "5" in reason


def test_build_ask_response_shape():
    payload = build_ask_response(
        answer="Answer text",
        sources=[],
        confidence_label="medium",
        confidence_reason="Answer based on 2 retrieved passages.",
        chunks_used=2,
    )
    assert payload["answer"] == "Answer text"
    assert payload["confidence"]["label"] == "medium"
    assert payload["metadata"]["chunks_used"] == 2
    assert payload["metadata"]["model"] == settings.GROQ_MODEL
