# RAG evaluation (free / local)

Lightweight checks that DocMind **retrieves sources** and content matches **expected keywords** from the sample report. No LangSmith, RAGAS, or paid services.

## Files

| File | Purpose |
|------|---------|
| `questions.json` | Eval cases tied to `sample_documents/ai-document-intelligence-report.pdf` |
| `scoring.py` | Keyword + source-count scoring (importable, unit-tested) |
| `run_eval.py` | CLI runner |

## Question schema

```json
{
  "id": "benefits",
  "question": "What are the main benefits?",
  "expected_keywords": ["benefits", "grounded"],
  "expected_min_sources": 1,
  "min_keyword_matches": 1,
  "match_in": "both"
}
```

| Field | Description |
|-------|-------------|
| `expected_keywords` | Terms that should appear in the answer and/or source previews |
| `expected_min_sources` | Minimum `sources[]` length from `/ask` |
| `min_keyword_matches` | How many keywords must match (default: 1) |
| `match_in` | `both` (default), `answer`, or `sources` |
| `require_all_keywords` | If `true`, every keyword must match |

## Run locally (retrieval only, **no Groq**)

Indexes the bundled sample in-process and scores **source previews** only:

```bash
python evals/run_eval.py --mode local
```

Requires project dependencies and will download the embedding model on first run.

## Run against API (full RAG + LLM)

Start the API with `GROQ_API_KEY`, then:

```bash
python evals/run_eval.py --mode api --api-url http://127.0.0.1:7860
```

This calls `POST /demo/load-sample` then `POST /ask` for each question.

Optional report file:

```bash
python evals/run_eval.py --mode api --output evals/results.json
```

Exit code `0` if all cases pass, `1` otherwise (usable in CI).

## Environment

| Variable | Used when |
|----------|-----------|
| `EVAL_API_URL` | Default base URL for `--mode api` |
