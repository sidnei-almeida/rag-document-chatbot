# Sample documents

Bundled PDFs for the public demo (`POST /demo/load-sample`).

| File | Purpose |
|------|---------|
| `ai-document-intelligence-report.pdf` | Default sample — technical overview of DocMind (benefits, limitations, stack). |

Override with environment variables:

- `SAMPLE_DOCUMENTS_DIR` — folder path (default: `sample_documents`)
- `SAMPLE_DOCUMENT_FILENAME` — PDF file name (default: `ai-document-intelligence-report.pdf`)

The `documents/` folder at the repo root is not used by the API automatically; add custom samples here and point the env vars to them if needed.
