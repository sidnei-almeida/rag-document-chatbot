# DocMind API - Endpoints Documentation

This document describes all available endpoints for the frontend development team.

**Base URLs**:
- Local development: `http://localhost:8000`
- Production (Hugging Face Spaces): `https://salmeida-my-rag-chatbot.hf.space`

---

## 1. Health Check

### GET `/health`

Check if the API is ready to receive requests.

**Request**: None

**Response**:
```json
{
  "status": "ok" | "initializing"
}
```

**Status Codes**:
- `200 OK`: API is operational

**Examples**:
```bash
# Local
curl http://localhost:8000/health
# Production
curl https://salmeida-my-rag-chatbot.hf.space/health
```

---

## 2. API Status & Info

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

## 3. Upload PDF Document

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

**Status Codes**:
- `200 OK`: PDF processed successfully
- `400 Bad Request`: File is not a PDF or missing file
- `500 Internal Server Error`: Error processing PDF

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

## 4. Ask Question

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
  "sources": [0, 1, 2]
}
```

**Fields**:
- `answer` (string): The generated answer based on the documents
- `sources` (array of numbers): Page numbers where the information was found

**Status Codes**:
- `200 OK`: Question answered successfully
- `400 Bad Request`: No documents indexed yet (need to upload PDF first)
- `503 Service Unavailable`: API is still initializing
- `500 Internal Server Error`: Error processing the question

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
- You must upload at least one PDF before asking questions
- The API searches the 3 most relevant document chunks
- Response time depends on document size and question complexity
- Sources array contains page numbers (0-indexed)

---

## Error Responses

All endpoints may return the following error formats:

### 400 Bad Request
```json
{
  "detail": "No documents indexed yet. Please upload a PDF first using /upload endpoint."
}
```

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

1. **Check API Status** (`GET /`)
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
    return { answer: data.answer, sources: data.sources };
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
5. **Sources**: Page numbers are 0-indexed (page 0 = first page)
6. **Multiple Uploads**: Users can upload multiple PDFs - they will be added to the same index

---

## Support

For questions or issues, contact the backend team or check the main README.md file.

