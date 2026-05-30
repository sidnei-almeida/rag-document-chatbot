# DocMind API — manual testing (workspace RAG)

Base URL (local): `http://localhost:7860`  
Production: `https://salmeida-my-rag-chatbot.hf.space`

Set `API` in your shell:

```bash
export API=http://localhost:7860
```

## 1. Health with no documents

```bash
curl -s "$API/health" | jq
```

Expected (before any upload):

- `api_ready`: true
- `llm_ready`: true
- `embeddings_ready`: true
- `storage_ready`: true
- `documents_ready`: false
- `workspace_count`: 0
- `default_workspace_id`: null

## 2. Upload one PDF (primary endpoint)

```bash
curl -s -X POST "$API/workspaces/upload" \
  -F "files=@document.pdf" | jq
```

Save `workspace_id` and `title` (should match filename for a single file).

## 3. Upload two PDFs in one workspace

```bash
curl -s -X POST "$API/workspaces/upload" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" | jq
```

Expected:

- `document_count`: 2
- `documents[].filename` for each file
- `title` like `doc1.pdf + doc2.pdf` or `doc1.pdf + 1 files` if long

## 4. List workspaces

```bash
curl -s "$API/workspaces" | jq
```

## 5. Ask workspace A

```bash
curl -s -X POST "$API/ask" \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"WS_A","question":"What is the document about?"}' | jq
```

Replace `WS_A` with the id from step 2.

Check `sources[].workspace_id`, `sources[].filename`, `sources[].document_id`.

## 6. Ask workspace B (isolation)

Upload another PDF (step 2 with a different file) → `WS_B`.

```bash
curl -s -X POST "$API/ask" \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"WS_B","question":"What is the document about?"}' | jq
```

**Isolation check:** sources in WS_A responses must only reference WS_A filenames; WS_B only WS_B.

## 7. Ask without workspace_id

```bash
curl -s -X POST "$API/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Hello"}' | jq
```

With no default workspace: HTTP 400 and `{"error":"No workspace_id provided. Upload one or more PDFs first."}`

## 8. Delete workspace

```bash
curl -s -X DELETE "$API/workspaces/WS_A" | jq
curl -s -X POST "$API/ask" \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":"WS_A","question":"What is this?"}' | jq
```

Expected: 404 after delete.

## 9. Clear all workspaces

```bash
curl -s -X DELETE "$API/clear" | jq
```

Or one workspace:

```bash
curl -s -X DELETE "$API/clear?workspace_id=WS_B" | jq
```

## 10. Legacy upload (compat)

```bash
curl -s -X POST "$API/upload" -F "file=@document.pdf" | jq
```

Response must include `workspace_id`, `title`, `documents`, `total_pages`, `total_chunks`, `index_ready`.

## Critical isolation scenario

1. Upload `financial.pdf` → note `workspace_id` **F**
2. Upload `medical.pdf` → note `workspace_id` **M**
3. Ask **F**: `"What is the document about?"` → sources only `financial.pdf`, `workspace_id` = F
4. Ask **M**: same question → sources only `medical.pdf`, `workspace_id` = M

If financial answers cite medical (or vice versa), isolation failed.
