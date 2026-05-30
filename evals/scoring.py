"""Pure scoring helpers for RAG evaluation (no external services)."""

from __future__ import annotations

from typing import Any


def count_keyword_matches(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in text (case-insensitive)."""
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lower)


def collect_match_text(result: dict[str, Any], match_in: str) -> str:
    """Build a single string from answer and/or source previews."""
    parts: list[str] = []

    if match_in in ("answer", "both"):
        parts.append(str(result.get("answer", "")))

    if match_in in ("sources", "both"):
        for source in result.get("sources") or []:
            if isinstance(source, dict):
                parts.append(str(source.get("preview", "")))
            else:
                parts.append(str(source))

    return " ".join(parts)


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """
    Score one eval case against an /ask-like result payload.

    Returns a dict with pass/fail and diagnostic fields.
    """
    question_id = case.get("id", case["question"][:40])
    sources = result.get("sources") or []
    min_sources = int(case.get("expected_min_sources", 1))
    keywords = list(case["expected_keywords"])
    min_matches = int(case.get("min_keyword_matches", 1))
    match_in = case.get("match_in", "both")
    require_all = bool(case.get("require_all_keywords", False))

    sources_ok = len(sources) >= min_sources
    match_text = collect_match_text(result, match_in)
    matched_count = count_keyword_matches(match_text, keywords)

    if require_all:
        keywords_ok = matched_count >= len(keywords)
    else:
        keywords_ok = matched_count >= min_matches

    no_evidence = "could not find enough evidence" in str(result.get("answer", "")).lower()

    passed = sources_ok and keywords_ok and not no_evidence
    reasons: list[str] = []
    if not sources_ok:
        reasons.append(f"sources={len(sources)} (need>={min_sources})")
    if not keywords_ok:
        reasons.append(
            f"keywords={matched_count} (need>={min_matches if not require_all else len(keywords)})"
        )
    if no_evidence:
        reasons.append("no-evidence answer")

    return {
        "id": question_id,
        "question": case["question"],
        "passed": passed,
        "sources_count": len(sources),
        "keyword_matches": matched_count,
        "match_in": match_in,
        "reason": "ok" if passed else "; ".join(reasons),
    }


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for row in rows if row["passed"])
    total = len(rows)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "all_passed": passed == total and total > 0,
    }
