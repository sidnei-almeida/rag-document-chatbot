# DocMind API - Endpoints Documentation

This document describes all available endpoints for the frontend development team.

**Base URLs**:
- Local development: `http://localhost:8000`
- Production (Hugging Face Spaces): `https://salmeida-my-rag-chatbot.hf.space`

---

## 1. Health Check

### GET `/health`

Lightweight health check with model, retrieval, and **public demo limits**.

**Response**:
```json
{
  "status": "ok",
  "api_ready": true,
  "llm_ready": true,
  "index_ready": true,
  "index_path": "/tmp/docmind_faiss_index/index.faiss",
  "model": "llama-3.3-70b-versatile",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "retrieval": {
    "type": "mmr",
    "k": 6,
    "fetch_k": 20,
    "lambda": 0.7
  },
  "limits": {
    "max_file_size_mb": 8,
    "max_pages": 40,
    "max_question_length": 1000
  }
}
```

**Examples**:
```bash
curl http://localhost:7860/health
```

---

## 2. Status (document / index)

### GET `/status`

Current index and document state for the demo UI.

**Response**:
```json
{
  "status": "ok",
  "index_ready": true,
  "document_loaded": true,
  "file_name": "report.pdf",
  "pages": 12,
  "chunks": 48,
  "model": "llama-3.3-70b-versatile",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "retrieval": { "type": "mmr", "k": 6 },
  "limits": {
    "max_file_size_mb": 8,
    "max_pages": 40,
    "max_question_length": 1000
  }
}
```

`pages` / `chunks` are set after the latest upload; they may be `null` if the index was loaded from disk only.

---

## 3. API Status & Info

### GET `/`

Get API status and available endpoints.

**Request**: None

**Response**:
```json
{
  "status": "DocMind API online",
  "endpoints": [
    "/ask (POST) - Ask questions about documents",
    "/upload (POST) - Upload PDF files to process"
  ],
  "health": "ok" | "initializing",
  "index_ready": true | false
}
```

**Fields**:
- `status`: API status message
- `endpoints`: List of available endpoints
- `health`: Current health status
- `index_ready`: Whether documents have been indexed (can ask questions)

**Status Codes**:
- `200 OK`: Success

**Examples**:
```bash
# Local
curl http://localhost:8000/
# Production
curl https://salmeida-my-rag-chatbot.hf.space/
```

---

## 4. Load sample document (demo)

### POST `/demo/load-sample`

Loads the bundled sample PDF from `sample_documents/` (default: `ai-document-intelligence-report.pdf`) using the same ingestion pipeline as `/upload`. Intended for a future **“Try sample document”** button in the UI.

**Request**: None

**Response**:
```json
{
  "status": "sample_loaded",
  "document": {
    "file_name": "ai-document-intelligence-report.pdf",
    "pages": 7,
    "chunks": 12
  },
  "index_ready": true
}
```

**Status Codes**:
- `200 OK`: Sample indexed successfully
- `404 Not Found`: Sample file missing on server (deploy without `sample_documents/`)
- `400 Bad Request`: Sample exceeds demo page limit
- `503 Service Unavailable`: API still initializing

**Example**:
```bash
curl -X POST http://localhost:7860/demo/load-sample
```

**Suggested questions after load**:
- What is this document about?
- What are the main benefits?
- What are the limitations?
- What technologies are used?
- Summarize the document.

**Configuration**: `SAMPLE_DOCUMENTS_DIR`, `SAMPLE_DOCUMENT_FILENAME` (see `.env.example`).

---

## 5. Upload PDF Document

### POST `/upload`

Upload a PDF file to be processed and indexed. This endpoint processes the PDF, creates embeddings, and updates the FAISS vector index.

**Content-Type**: `multipart/form-data`

**Request Body**:
- `file` (File, required): PDF file to upload

**Response**:
```json
{
  "message": "PDF processed successfully",
  "filename": "document.pdf",
  "pages": 10,
  "chunks": 45,
  "status": "ready"
}
```

**Fields**:
- `message`: Success message
- `filename`: Name of the uploaded file
- `pages`: Number of pages in the PDF
- `chunks`: Number of text chunks created for indexing
- `status`: Processing status

**Demo limits** (configurable via environment variables):

| Limit | Default | Error message (400) |
|-------|---------|---------------------|
| File size | 8 MB | `PDF too large for this public demo.` |
| Pages | 40 | `This demo supports PDFs up to 40 pages.` |
| Extension | `.pdf` only | `Only PDF files are supported.` |

**Status Codes**:
- `200 OK`: PDF processed successfully
- `400 Bad Request`: Invalid file, too large, or too many pages
- `500 Internal Server Error`: Processing failed (generic message, no raw traceback)

**Example with curl**:
```bash
# Local
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"

# Production
curl -X POST https://salmeida-my-rag-chatbot.hf.space/upload \
  -F "file=@document.pdf"
```

**Example with JavaScript (Fetch)**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data);
```

**Example with JavaScript (Axios)**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await axios.post('http://localhost:8000/upload', formData, {
  headers: {
    'Content-Type': 'multipart/form-data'
  }
});

console.log(response.data);
```

**Notes**:
- Only PDF files are accepted
- The first upload creates a new index
- Subsequent uploads add to the existing index
- Processing may take a few seconds depending on PDF size

---

## 6. Ask Question

### POST `/ask`

Ask a question about the uploaded documents. The API will search the indexed documents and generate an answer using the Groq LLM.

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "question": "What is the main theme of the document?"
}
```

**Fields**:
- `question` (string, required): The question to ask about the documents

**Response**:
```json
{
  "answer": "The main theme of the document is about...",
  "sources": [
    {
      "chunk_id": 0,
      "page": 1,
      "page_index": 0,
      "file_name": "document.pdf",
      "preview": "Short excerpt from the retrieved chunk...",
      "score": 0.42
    }
  ],
  "confidence": {
    "label": "high",
    "reason": "Answer based on 5 retrieved passages."
  },
  "metadata": {
    "model": "llama-3.3-70b-versatile",
    "retrieval_k": 6,
    "chunks_used": 5,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

**Fields**:
- `answer` (string): The generated answer based on the documents
- `sources` (array): Structured evidence for each retrieved chunk
  - `chunk_id`: Chunk identifier (index or metadata id)
  - `page`: **1-indexed** page number for display (never use 0 for end users)
  - `page_index`: **0-indexed** page from PDF metadata (PyPDFLoader default)
  - `file_name`: Source PDF file name
  - `preview`: Cleaned excerpt (~320 characters)
  - `score`: Similarity distance when available; `null` when using MMR retriever fallback
- `confidence` (object): Heuristic confidence (`high` | `medium` | `low`) and short `reason`
- `metadata` (object): `model`, `retrieval_k`, `chunks_used`, `embedding_model`

**No-evidence response** (no LLM call; empty retrieval):
```json
{
  "answer": "I could not find enough evidence in the uploaded document to answer that reliably.",
  "sources": [],
  "confidence": {
    "label": "low",
    "reason": "No reliable document context was found."
  },
  "metadata": {
    "model": "llama-3.3-70b-versatile",
    "retrieval_k": 6,
    "chunks_used": 0,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

**Question validation**:

| Rule | Error (400) |
|------|----------------|
| Empty / whitespace only | `Question cannot be empty.` |
| Longer than `MAX_QUESTION_LENGTH` (default 1000) | `Question is too long for this demo.` |
| No index (non-general) | `503` — RAG disabled until index is ready (use upload or load-sample) |

**Status Codes**:
- `200 OK`: Question answered successfully
- `400 Bad Request`: Validation errors only
- `429 Too Many Requests`: Groq rate limit (friendly message)
- `503 Service Unavailable`: API initializing, LLM missing, or **RAG index not ready**
- `500 Internal Server Error`: Generic processing error

---

## 7. Clear Index

### DELETE `/clear`

Clears in-memory FAISS state, sets `index_cleared` so the old disk index is **not** reloaded automatically, and attempts to remove the local `faiss_index` folder.

**Response**:
```json
{
  "status": "cleared",
  "index_ready": false,
  "message": "Index cleared successfully."
}
```

**Example with curl**:
```bash
# Local
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main theme of the document?"}'

# Production
curl -X POST https://salmeida-my-rag-chatbot.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main theme of the document?"}'
```

**Example with JavaScript (Fetch)**:
```javascript
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    question: 'What is the main theme of the document?'
  })
});

const data = await response.json();
console.log(data.answer);
console.log('Sources:', data.sources);
```

**Example with JavaScript (Axios)**:
```javascript
const response = await axios.post('http://localhost:8000/ask', {
  question: 'What is the main theme of the document?'
});

console.log(response.data.answer);
console.log('Sources:', response.data.sources);
```

**Notes**:
- You must upload at least one PDF before asking questions (except general greetings like "hello")
- Retrieval uses **MMR** by default; chunk count is configurable via `RETRIEVAL_K` (default **6**)
- When possible, retrieval uses `similarity_search_with_score` so `sources[].score` is populated
- Use `page` (1-indexed) in the UI; use `page_index` only for internal/debug mapping
- `confidence` is heuristic (chunk count), not a separate ML model
- If no relevant chunks are found, the API returns an honest message and **does not** call the LLM

---

## Error Responses

All endpoints may return the following error formats:

### 400 Bad Request
```json
{
  "detail": "Please upload a PDF before asking questions."
}
```

Other examples: `PDF too large for this public demo.`, `Question is too long for this demo.`

### 500 Internal Server Error
```json
{
  "detail": "Error processing question: ErrorType: error message"
}
```

### 503 Service Unavailable
```json
{
  "detail": "API is still initializing."
}
```

---

## Frontend Integration Flow

### Recommended User Flow:

1. **Check API Status** (`GET /health` or `GET /status`)
   - Display loading state if `health: "initializing"`
   - Show upload button if `index_ready: false`
   - Show question form if `index_ready: true`

2. **Upload PDF** (`POST /upload`)
   - Show file input
   - Display upload progress
   - Show success message with pages/chunks count
   - Enable question form after successful upload

3. **Ask Questions** (`POST /ask`)
   - Show question input form
   - Display loading state while processing
   - Show answer and sources
   - Handle errors gracefully

### Example React Component Structure:

```javascript
// Check if index is ready
const checkStatus = async () => {
  const response = await fetch('/');
  const data = await response.json();
  setIndexReady(data.index_ready);
};

// Upload PDF
const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/upload', {
    method: 'POST',
    body: formData
  });
  
  if (response.ok) {
    const data = await response.json();
    setIndexReady(true);
    alert(`PDF processed! ${data.pages} pages, ${data.chunks} chunks`);
  }
};

// Ask question
const askQuestion = async (question) => {
  const response = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  
  if (response.ok) {
    const data = await response.json();
    return {
      answer: data.answer,
      sources: data.sources,
      confidence: data.confidence,
      metadata: data.metadata,
    };
  } else {
    const error = await response.json();
    throw new Error(error.detail);
  }
};
```

---

## Testing Endpoints

### Using Swagger UI (Interactive Documentation)

Visit `http://localhost:8000/docs` in your browser to access the interactive API documentation with a built-in testing interface.

### Using Postman

1. Import the endpoints
2. Set base URL to `http://localhost:8000`
3. Test each endpoint with the examples above

---

## Notes for Frontend Developers

1. **CORS**: If deploying to different domains, ensure CORS is configured on the backend
2. **File Size**: Consider file size limits for PDF uploads (recommend max 10MB)
3. **Loading States**: Processing can take 5-30 seconds depending on PDF size
4. **Error Handling**: Always check response status and display user-friendly error messages
5. **Sources**: Display `page` (1-indexed). Use `page_index` only if you need the raw metadata value.
6. **Retrieval**: Configure `RETRIEVAL_K`, `RETRIEVAL_FETCH_K`, and `RETRIEVAL_LAMBDA` via environment variables (see `.env.example`).
7. **CORS**: Set `ALLOWED_ORIGINS=*` (default) or a comma-separated list of origins.
8. **Demo limits**: `MAX_FILE_SIZE_MB`, `MAX_PAGES`, `MAX_QUESTION_LENGTH` keep the public Space stable and free-tier friendly.
9. **Multiple Uploads**: Users can upload multiple PDFs - they will be added to the same index unless `replace=true`.

---

## Support

For questions or issues, contact the backend team or check the main README.md file.

