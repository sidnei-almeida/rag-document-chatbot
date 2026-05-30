"""Unit tests for RAG retrieval helpers (no external services)."""

from langchain_core.documents import Document

from app.core.config import settings
from app.services.retrieval import (
    build_ask_response,
    clean_preview,
    compute_confidence,
    extract_page_index,
    format_sources_for_workspace,
)


def test_extract_page_index_defaults_to_zero():
    assert extract_page_index({}) == 0
    assert extract_page_index({"page": 3}) == 3


def test_page_is_one_indexed_in_sources():
    docs = [
        Document(
            page_content="Revenue grew in Q4.",
            metadata={
                "workspace_id": "ws_test",
                "document_id": "doc_1",
                "filename": "report.pdf",
                "page": 0,
                "chunk_id": "chunk_007",
            },
        )
    ]
    sources = format_sources_for_workspace("ws_test", docs, scores=[0.25])
    assert len(sources) == 1
    assert sources[0]["page"] == 1
    assert sources[0]["filename"] == "report.pdf"
    assert sources[0]["chunk_id"] == "chunk_007"
    assert sources[0]["workspace_id"] == "ws_test"
    assert sources[0]["score"] == 0.25


def test_format_sources_discards_wrong_workspace():
    docs = [
        Document(
            page_content="Secret from B",
            metadata={
                "workspace_id": "ws_b",
                "document_id": "doc_b",
                "filename": "b.pdf",
                "page": 0,
                "chunk_id": "chunk_000",
            },
        ),
        Document(
            page_content="Valid from A",
            metadata={
                "workspace_id": "ws_a",
                "document_id": "doc_a",
                "filename": "a.pdf",
                "page": 0,
                "chunk_id": "chunk_001",
            },
        ),
    ]
    sources = format_sources_for_workspace("ws_a", docs)
    assert len(sources) == 1
    assert sources[0]["filename"] == "a.pdf"


def test_clean_preview_collapses_whitespace_and_truncates():
    long_text = "word " * 200
    preview = clean_preview(f"  line1\n\n  line2   {long_text}")
    assert "\n" not in preview
    assert "  " not in preview
    assert len(preview) <= settings.PREVIEW_MAX_LENGTH


def test_format_sources_skips_empty_chunks():
    docs = [
        Document(
            page_content="   ",
            metadata={"workspace_id": "ws", "document_id": "d", "filename": "f.pdf", "page": 0},
        ),
        Document(
            page_content="Valid content.",
            metadata={
                "workspace_id": "ws",
                "document_id": "d",
                "filename": "f.pdf",
                "page": 1,
                "chunk_id": "chunk_001",
            },
        ),
    ]
    sources = format_sources_for_workspace("ws", docs)
    assert len(sources) == 1
    assert sources[0]["page"] == 2


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
        workspace_id="ws_abc",
        answer="Answer text",
        sources=[],
        confidence_label="medium",
        confidence_reason="Based on context.",
        retrieval_used=True,
        latency_ms=100,
    )
    assert payload["workspace_id"] == "ws_abc"
    assert payload["confidence"] == "medium"
    assert payload["retrieval_used"] is True
    assert payload["latency_ms"] == 100
    assert payload["model"] == settings.GROQ_MODEL
