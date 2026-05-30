"""Unit tests for eval scoring (no API, no Groq)."""

from evals.scoring import count_keyword_matches, evaluate_case, summarize_results


def test_count_keyword_matches():
    assert count_keyword_matches("DocMind uses FAISS", ["faiss", "pinecone"]) == 1
    assert count_keyword_matches("hello", ["world"]) == 0


def test_evaluate_case_passes_with_sources_and_keywords():
    case = {
        "id": "tech",
        "question": "What technologies are used?",
        "expected_keywords": ["fastapi", "faiss"],
        "expected_min_sources": 1,
        "min_keyword_matches": 2,
        "match_in": "sources",
    }
    result = {
        "answer": "",
        "sources": [
            {
                "preview": "Backend FastAPI and FAISS vector store.",
                "page": 1,
            }
        ],
    }
    row = evaluate_case(case, result)
    assert row["passed"] is True
    assert row["sources_count"] == 1
    assert row["keyword_matches"] >= 2


def test_evaluate_case_fails_without_sources():
    case = {
        "id": "x",
        "question": "Q?",
        "expected_keywords": ["docmind"],
        "expected_min_sources": 1,
    }
    row = evaluate_case(case, {"answer": "text", "sources": []})
    assert row["passed"] is False
    assert "sources=0" in row["reason"]


def test_evaluate_case_fails_on_no_evidence_answer():
    case = {
        "id": "x",
        "question": "Q?",
        "expected_keywords": ["docmind"],
        "expected_min_sources": 0,
    }
    result = {
        "answer": "I could not find enough evidence in the uploaded document to answer that reliably.",
        "sources": [],
    }
    row = evaluate_case(case, result)
    assert row["passed"] is False
    assert "no-evidence" in row["reason"]


def test_summarize_results():
    rows = [{"passed": True}, {"passed": False}]
    summary = summarize_results(rows)
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["all_passed"] is False
