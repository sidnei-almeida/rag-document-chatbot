#!/usr/bin/env python3
"""
Run a simple offline RAG evaluation using evals/questions.json.

Modes:
  local  — index the bundled sample in-process; score retrieval (sources + previews).
           Does not call Groq (no API key required).
  api    — call a running DocMind API (/demo/load-sample + /ask). Requires Groq on the server.

Examples:
  python evals/run_eval.py --mode local
  python evals/run_eval.py --mode api --api-url http://127.0.0.1:7860
  python evals/run_eval.py --mode api --api-url http://127.0.0.1:7860 --output evals/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_questions(path: Path | None = None) -> list[dict]:
    path = path or QUESTIONS_PATH
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    return data


def run_local_eval(questions: list[dict]) -> list[dict]:
    """Index sample PDF and evaluate retrieval without calling the LLM."""
    os.environ.setdefault("GROQ_API_KEY", "eval-local-no-llm")
    sys.path.insert(0, str(PROJECT_ROOT))

    from langchain_huggingface import HuggingFaceEmbeddings

    from app.core.config import settings
    from app.core.state import state
    from app.services.retrieval import (
        filter_useful_documents_with_scores,
        format_sources,
        retrieve_documents,
    )
    from app.services.sample_loader import load_sample_document

    print("--> Loading embeddings (one-time download possible)...")
    state.embeddings_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    print("--> Indexing sample document...")
    chunks, pages, name = load_sample_document()
    print(f"    {name}: {pages} pages, {chunks} chunks")

    rows: list[dict] = []
    from evals.scoring import evaluate_case

    for case in questions:
        docs, scores = retrieve_documents(case["question"])
        useful, aligned = filter_useful_documents_with_scores(docs, scores)
        sources = format_sources(useful, aligned)
        match_in = case.get("match_in", "sources")
        result = {
            "answer": "",
            "sources": sources,
            "confidence": {"label": "n/a", "reason": "local retrieval-only eval"},
        }
        row = evaluate_case(case, result)
        if match_in != case.get("match_in"):
            row["note"] = f"forced match_in={match_in} for local mode"
        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"  [{status}] {row['id']}: {row['reason']}")

    return rows


def run_api_eval(questions: list[dict], api_url: str, load_sample: bool) -> list[dict]:
    """Evaluate via HTTP against a running DocMind instance."""
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Install httpx: pip install httpx") from exc

    from evals.scoring import evaluate_case

    base = api_url.rstrip("/")
    rows: list[dict] = []

    with httpx.Client(base_url=base, timeout=120.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print(f"--> API health: {health.json().get('status')}")

        if load_sample:
            print("--> POST /demo/load-sample")
            sample = client.post("/demo/load-sample")
            sample.raise_for_status()
            meta = sample.json()
            doc = meta.get("document", {})
            print(
                f"    loaded {doc.get('file_name')}: "
                f"{doc.get('pages')} pages, {doc.get('chunks')} chunks"
            )

        for case in questions:
            response = client.post("/ask", json={"question": case["question"]})
            if response.status_code != 200:
                rows.append(
                    {
                        "id": case.get("id", "?"),
                        "question": case["question"],
                        "passed": False,
                        "sources_count": 0,
                        "keyword_matches": 0,
                        "match_in": case.get("match_in", "both"),
                        "reason": f"HTTP {response.status_code}: {response.text[:200]}",
                    }
                )
                print(f"  [FAIL] {case.get('id')}: HTTP {response.status_code}")
                continue

            result = response.json()
            row = evaluate_case(case, result)
            rows.append(row)
            status = "PASS" if row["passed"] else "FAIL"
            print(f"  [{status}] {row['id']}: {row['reason']}")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple DocMind RAG evaluation")
    parser.add_argument(
        "--mode",
        choices=("local", "api"),
        default="local",
        help="local=retrieval-only (no Groq); api=full /ask against running server",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("EVAL_API_URL", "http://127.0.0.1:7860"),
        help="Base URL when --mode api",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_PATH,
        help="Path to questions JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )
    parser.add_argument(
        "--skip-load-sample",
        action="store_true",
        help="API mode only: skip POST /demo/load-sample",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} eval questions from {args.questions}")

    if args.mode == "local":
        rows = run_local_eval(questions)
    else:
        rows = run_api_eval(questions, args.api_url, load_sample=not args.skip_load_sample)

    from evals.scoring import summarize_results

    summary = summarize_results(rows)
    print()
    print(
        f"Summary: {summary['passed']}/{summary['total']} passed "
        f"(pass_rate={summary['pass_rate']})"
    )

    report = {"summary": summary, "results": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
