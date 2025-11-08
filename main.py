import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

# --- LangChain & AI ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
VECTOR_STORE_NAME = "faiss_index"
# AI model to use via Groq
# Available models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768 (deprecated)
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Global Variables (load only once) ---
vector_store = None
retriever = None
llm = None
embeddings_model = None

class QuestionRequest(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global vector_store, retriever, llm, embeddings_model
    print("--> Initializing API...")

    # 1. Check if API key is configured
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not configured! Set the GROQ_API_KEY environment variable.")
        raise ValueError("GROQ_API_KEY not configured. Set the GROQ_API_KEY environment variable.")

    # 2. Load the same Embeddings model used during ingestion
    print("--> Loading Embeddings model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Load the FAISS vector database (if it exists)
    print(f"--> Loading FAISS vector database from '{VECTOR_STORE_NAME}'...")
    try:
        if os.path.exists(VECTOR_STORE_NAME):
            vector_store = FAISS.load_local(
                VECTOR_STORE_NAME, 
                embeddings_model, 
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            print("    Vector database loaded successfully!")
        else:
            print(f"    No existing vector database found. Waiting for PDF upload...")
            # Create empty vector store to avoid errors
            vector_store = None
            retriever = None
    except Exception as e:
        print(f"WARNING: Could not load folder {VECTOR_STORE_NAME}.")
        print(f"Error: {str(e)}")
        print("You can upload a PDF to create the index.")
        vector_store = None
        retriever = None

    # 4. Configure LLM using Groq (fast and reliable)
    print(f"--> Connecting to Groq model ({GROQ_MODEL})...")
    # Set environment variable for Groq to use
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    try:
        llm = ChatGroq(
            model_name=GROQ_MODEL,
            temperature=0.3,
            max_tokens=512,
        )
        print("    Groq LLM configured successfully!")
    except Exception as e:
        print(f"ERROR configuring LLM: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        raise e
    
    print("--> API READY FOR USE!")
    
    yield
    
    # Shutdown (if needed)
    print("--> Shutting down API...")

app = FastAPI(
    title="DocMind API", 
    description="RAG Chatbot with FastAPI and Docker",
    lifespan=lifespan
)

def create_prompt(context: str, question: str) -> str:
    """Creates a simple prompt combining context and question"""
    prompt = f"""Use the following context to answer the question.
If you don't know the answer, just say you don't know, don't make up an answer.

Context:
{context}

Question: {question}

Answer:"""
    return prompt

@app.post("/ask")
async def ask_document(req: QuestionRequest):
    global retriever, llm
    
    if not llm:
        raise HTTPException(status_code=503, detail="API is still initializing.")
    
    if not retriever:
        raise HTTPException(
            status_code=400, 
            detail="No documents indexed yet. Please upload a PDF first using /upload endpoint."
        )
    
    try:
        print(f"Receiving question: {req.question}")
        
        # 1. Search for relevant documents in FAISS (manual)
        print("--> Searching for relevant documents in FAISS...")
        docs = retriever.invoke(req.question)
        
        # 2. Format context from found documents
        context = "\n\n".join([doc.page_content for doc in docs])
        print(f"    Found {len(docs)} relevant documents")
        
        # 3. Create simple prompt (manual)
        prompt = create_prompt(context, req.question)
        
        # 4. Send prompt directly to Groq LLM (without chains)
        print("--> Sending prompt to Groq LLM...")
        # ChatGroq returns a Message object, we need to extract the content
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 5. Extract sources (pages) from documents
        sources = [doc.metadata.get("page", "Unknown") for doc in docs]
        
        print("--> Response generated successfully!")
        
        return {
            "answer": response_text,
            "sources": sources
        }
    except Exception as e:
        import traceback
        error_details = str(e)
        error_type = type(e).__name__
        traceback_str = traceback.format_exc()
        print(f"Error processing question: {error_details}")
        print(f"Type: {error_type}")
        print(f"Full traceback:\n{traceback_str}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {error_type}: {error_details}"
        )

def process_pdf_and_update_index(pdf_path: str):
    """Process PDF file and update FAISS index"""
    global vector_store, retriever, embeddings_model
    
    print(f"--> Processing PDF: {pdf_path}")
    
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"    PDF loaded. Total pages read: {len(documents)}")
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=400
    )
    chunks = text_splitter.split_documents(documents)
    print(f"    Document split into {len(chunks)} chunks")
    
    # 3. Create or update FAISS index
    if vector_store is None:
        # Create new index
        print("--> Creating new FAISS index...")
        vector_store = FAISS.from_documents(chunks, embeddings_model)
    else:
        # Add to existing index
        print("--> Adding documents to existing FAISS index...")
        vector_store.add_documents(chunks)
    
    # 4. Save index to disk
    vector_store.save_local(VECTOR_STORE_NAME)
    print(f"    Index saved to '{VECTOR_STORE_NAME}'")
    
    # 5. Update retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    
    return len(chunks), len(documents)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file to update the FAISS index"""
    global vector_store, retriever
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create temporary file to save uploaded PDF
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Save uploaded file to temporary location
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        print(f"--> Received PDF upload: {file.filename}")
        
        # Process PDF and update index
        chunks_count, pages_count = process_pdf_and_update_index(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "pages": pages_count,
            "chunks": chunks_count,
            "status": "ready"
        }
        
    except Exception as e:
        import traceback
        error_details = str(e)
        error_type = type(e).__name__
        traceback_str = traceback.format_exc()
        print(f"Error processing PDF: {error_details}")
        print(f"Type: {error_type}")
        print(f"Full traceback:\n{traceback_str}")
        
        # Clean up temporary file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {error_type}: {error_details}"
        )

@app.get("/")
def home():
    return {
        "status": "DocMind API online", 
        "endpoints": [
            "/ask (POST) - Ask questions about documents",
            "/upload (POST) - Upload PDF files to process"
        ],
        "health": "ok" if (retriever and llm) else "initializing",
        "index_ready": retriever is not None
    }

@app.get("/health")
def health():
    return {"status": "ok" if (retriever and llm) else "initializing"}
