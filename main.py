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
AGENT_PERSONALITY = ""
PERSONALITY_FILE = "AGENT_PERSONALITY.txt"

def load_personality():
    """Load personality from file"""
    global AGENT_PERSONALITY
    try:
        if os.path.exists(PERSONALITY_FILE):
            with open(PERSONALITY_FILE, 'r', encoding='utf-8') as f:
                AGENT_PERSONALITY = f.read().strip()
            print(f"--> Agent personality loaded from {PERSONALITY_FILE}")
        else:
            print(f"WARNING: {PERSONALITY_FILE} not found, using default personality")
            AGENT_PERSONALITY = "You are a helpful and knowledgeable assistant."
    except Exception as e:
        print(f"Error loading personality: {e}")
        AGENT_PERSONALITY = "You are a helpful and knowledgeable assistant."

class QuestionRequest(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global vector_store, retriever, llm, embeddings_model
    print("--> Initializing API...")

    # 0. Load agent personality
    load_personality()

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
            temperature=0.7,  # Increased for more personality and creativity
            max_tokens=1024,  # Increased for more complete responses
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

def create_prompt(context: str, question: str, has_relevant_context: bool = True, documents_available: bool = True) -> str:
    """Creates a prompt combining personality, context and question"""
    
    # Check if question is general/conversational (doesn't need document context)
    general_questions = ["hello", "hi", "hey", "how are you", "what can you do", 
                        "help", "thanks", "thank you", "bye", "goodbye", "olá", "oi"]
    is_general = any(gq in question.lower() for gq in general_questions)
    
    if is_general:
        # For general questions, respond with personality but don't require context
        prompt = f"""{AGENT_PERSONALITY}

The user asked: {question}

Respond naturally and engagingly. You don't need to reference any document for this question. Be yourself!"""
    elif not documents_available:
        # No documents uploaded yet
        prompt = f"""{AGENT_PERSONALITY}

The user asked: {question}

Respond naturally. Documents are not uploaded yet."""
    elif not has_relevant_context:
        # Documents are available but the specific context found wasn't very relevant
        # Still use what we found, but acknowledge it might be limited
        if context and len(context) > 0:
            prompt = f"""{AGENT_PERSONALITY}

You have access to documents, but the specific context found for this question is limited. Use what you can from the context below, and answer based on your knowledge if needed.

Context from document:
{context}

Question: {question}

Answer based on the context above if relevant, but keep your unique voice and style:"""
        else:
            prompt = f"""{AGENT_PERSONALITY}

The user asked: {question}

You have access to documents, but couldn't find specific context for this question. Answer based on your general knowledge while maintaining your personality."""
    else:
        # For document-specific questions with good context, use it
        prompt = f"""{AGENT_PERSONALITY}

You have access to the following context from a document. Use it to answer the question, but maintain your personality and style.

Context from document:
{context}

Question: {question}

Answer based on the context above, but keep your unique voice and style:"""
    
    return prompt

@app.post("/ask")
async def ask_document(req: QuestionRequest):
    global retriever, llm
    
    if not llm:
        raise HTTPException(status_code=503, detail="API is still initializing.")
    
    try:
        print(f"Receiving question: {req.question}")
        
        # Check if question is general/conversational (doesn't need document context)
        general_questions = ["hello", "hi", "hey", "how are you", "what can you do", 
                            "help", "thanks", "thank you", "bye", "goodbye", "olá", "oi"]
        is_general = any(gq in req.question.lower() for gq in general_questions)
        
        # If no retriever but it's a general question, allow response
        if not retriever:
            if is_general:
                # Allow general questions even without documents
                print("--> General question detected, responding without document context...")
                prompt = create_prompt("", req.question, has_relevant_context=False, documents_available=False)
                response = llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                return {
                    "answer": response_text,
                    "sources": None
                }
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="No documents indexed yet. Please upload a PDF first using /upload endpoint, or ask a general question like 'Hello' or 'What can you do?'"
                )
        
        # 1. Search for relevant documents in FAISS (manual)
        print("--> Searching for relevant documents in FAISS...")
        docs = retriever.invoke(req.question)
        
        # 2. Format context from found documents
        context = "\n\n".join([doc.page_content for doc in docs])
        print(f"    Found {len(docs)} relevant documents")
        
        # Check if context is actually relevant (not just random chunks)
        # If similarity is too low or context is too short, treat as general question
        # Lowered threshold from 200 to 50 to be less restrictive
        has_relevant_context = len(context) > 50 and len(docs) > 0
        
        # 3. Create prompt with personality
        # Documents are available (retriever exists), even if specific context isn't very relevant
        prompt = create_prompt(context, req.question, has_relevant_context, documents_available=True)
        
        # 4. Send prompt directly to Groq LLM (without chains)
        print("--> Sending prompt to Groq LLM...")
        # ChatGroq returns a Message object, we need to extract the content
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 5. Extract sources (pages) from documents (only if relevant context was used)
        sources = []
        if has_relevant_context:
            sources = [doc.metadata.get("page", "Unknown") for doc in docs]
        
        print("--> Response generated successfully!")
        
        return {
            "answer": response_text,
            "sources": sources if sources else None
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
