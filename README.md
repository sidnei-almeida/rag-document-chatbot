---
title: RAG Document Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# RAG Document Chatbot

A serverless RAG (Retrieval-Augmented Generation) chatbot built with FastAPI, LangChain, and Docker. Hosted on Hugging Face Spaces to answer questions about PDF documents using open-source LLMs via Groq.

## 🚀 Features

- **PDF Processing**: Extracts and processes PDF documents
- **Vector Search**: Uses FAISS for efficient semantic search
- **RAG (Retrieval-Augmented Generation)**: Combines vector search with LLMs for accurate answers
- **REST API**: FastAPI interface for easy integration
- **Hugging Face Spaces Deployment**: Ready for deployment with Docker

## 📋 Prerequisites

- Python 3.11+
- Groq API Key (for using LLM models)
- Docker (for deployment)

## 🛠️ Local Installation

1. **Clone the repository**:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/rag-document-chatbot
cd rag-document-chatbot
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
# Install PyTorch CPU first (saves space)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
pip install -r requirements.txt
```

4. **Configure the Groq API key**:
```bash
export GROQ_API_KEY="your_api_key_here"
```

## 📚 Usage

### 1. Prepare Data (Ingestion)

Place your PDF file in the project root with the name `documento.pdf` and run:

```bash
python data_injector.py
```

This will:
- Load the PDF
- Split into chunks
- Create embeddings
- Save the FAISS index in `faiss_index/`

### 2. Run the API

```bash
python app.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Ask Questions

**Endpoint**: `POST /ask`

**Example with curl**:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main theme of the document?"}'
```

**Response**:
```json
{
  "answer": "The main theme of the document is...",
  "sources": [0, 1, 2]
}
```

**Other endpoints**:
- `GET /` - API status
- `GET /health` - Health check
- `GET /docs` - Interactive documentation (Swagger UI)

## 🐳 Deploy to Hugging Face Spaces

### Step by Step

1. **Create a new Space on Hugging Face**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Docker" as SDK
   - Give your Space a name

2. **Clone the Space repository**:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
cd SPACE_NAME
```

3. **Copy necessary files**:
```bash
cp ../rag-document-chatbot/* .
```

4. **Configure the Groq API key**:
   - In your Space, go to "Settings" → "Repository secrets"
   - Add a secret named `GROQ_API_KEY` with your API key

5. **Commit and push**:
```bash
git add .
git commit -m "Initial commit"
git push
```

Hugging Face Spaces will automatically:
- Build the Docker image
- Install dependencies
- Start the application on port 7860

### Required Files for Deployment

- `Dockerfile` - Docker container configuration
- `app.py` - Application entrypoint
- `main.py` - Main API code
- `requirements.txt` - Python dependencies
- `data_injector.py` - Ingestion script (optional in deployment)
- `faiss_index/` - Vector index (should be committed or generated during build)
- `.dockerignore` - Files to ignore in build

## 📁 Project Structure

```
rag-document-chatbot/
├── main.py              # Main FastAPI API
├── app.py               # Entrypoint for Hugging Face Spaces
├── data_injector.py     # Script to process PDFs and create index
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── .dockerignore       # Files ignored in Docker
├── README.md           # This file
├── documento.pdf       # Example PDF (not committed)
└── faiss_index/        # FAISS vector index
    ├── index.faiss
    └── index.pkl
```

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Groq API key (required)
- `VECTOR_STORE_NAME`: FAISS index folder name (default: `faiss_index`)
- `GROQ_MODEL`: LLM model to use (default: `llama-3.3-70b-versatile`)

### Supported Models

The project uses models via Groq API. You can change the model in `main.py`:

```python
GROQ_MODEL = "llama-3.3-70b-versatile"
```

Other available models:
- `llama-3.3-70b-versatile` - Recommended, high quality
- `llama-3.1-8b-instant` - Faster, smaller
- `llama-3.1-70b-versatile` - Alternative to 3.3
- `gemma2-9b-it` - Lightweight alternative

## 🐛 Troubleshooting

### Error: "GROQ_API_KEY not configured"
Set the `GROQ_API_KEY` environment variable with your Groq API key.

### Error: "Could not load folder faiss_index"
Run `python data_injector.py` first to create the index.

### Error: "ModuleNotFoundError"
Make sure to install all dependencies:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Deployment error: "Out of memory"
Hugging Face Spaces has memory limits. Consider:
- Using a smaller model
- Reducing FAISS index size
- Using `faiss-cpu` instead of `faiss-gpu`

### Model decommissioned error
Some models may be deprecated. Check Groq's documentation for current available models and update `GROQ_MODEL` in `main.py`.

## 📝 License

This project is open-source and available under the MIT license.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests.

## 📧 Contact

For questions or support, open an issue in the repository.
