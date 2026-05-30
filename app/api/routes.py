"""DocMind API route handlers."""

import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from groq import RateLimitError

from app.core.config import settings
from app.core.state import state
from app.prompts.rag import create_general_prompt, create_rag_prompt, is_general_question
from app.schemas.ask import AskResponse, QuestionRequest
from app.schemas.demo import SampleDocumentInfo, SampleLoadResponse
from app.api.rag_guard import require_api_ready, require_llm_ready, require_rag_index
from app.services.sample_loader import load_sample_document
from app.api.validators import validate_pdf_upload, validate_question
from app.services.ingestion import clear_vector_index, process_pdf_and_update_index
from app.services.retrieval import (
    build_ask_response,
    compute_confidence,
    filter_useful_documents_with_scores,
    format_context_from_documents,
    format_sources,
    retrieve_documents,
)
from app.services.status import build_health_payload, build_status_payload

logger = logging.getLogger("docmind")
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_document(req: QuestionRequest):
    try:
        question = validate_question(req.question)
        logger.info("Received question (%s chars)", len(question))

        general = is_general_question(question)

        if general:
            require_llm_ready()
            logger.info("General question with index present; skipping retrieval")
            prompt = create_general_prompt(question)
            response = state.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            return build_ask_response(
                answer=answer,
                sources=[],
                confidence_label="medium",
                confidence_reason="General conversational question; no document retrieval performed.",
            )

        require_rag_index()
        logger.info("Searching FAISS for relevant chunks")
        docs, scores = retrieve_documents(question)
        useful_docs, aligned_scores = filter_useful_documents_with_scores(docs, scores)
        logger.info("Found %s chunks (%s usable)", len(docs), len(useful_docs))

        if not useful_docs:
            logger.info("No relevant chunks; returning no-evidence response")
            return build_ask_response(
                answer=settings.NO_EVIDENCE_MESSAGE,
                sources=[],
                confidence_label="low",
                confidence_reason="No reliable document context was found.",
            )

        structured_sources = format_sources(useful_docs, aligned_scores)
        confidence_label, confidence_reason = compute_confidence(
            len(structured_sources), has_context=True, scores=aligned_scores
        )
        context = format_context_from_documents(useful_docs)
        prompt = create_rag_prompt(question, context)

        logger.info("Sending prompt to Groq LLM")
        response = state.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info("Answer generated successfully")

        return build_ask_response(
            answer=answer,
            sources=structured_sources,
            confidence_label=confidence_label,
            confidence_reason=confidence_reason,
            chunks_used=len(structured_sources),
        )
    except HTTPException:
        raise
    except RateLimitError as exc:
        error_details = getattr(exc, "message", str(exc))
        logger.warning("Groq rate limit: %s", error_details)
        raise HTTPException(
            status_code=429,
            detail="The AI service is temporarily busy. Please try again in a moment.",
        ) from exc
    except Exception:
        logger.exception("Error processing question")
        raise HTTPException(
            status_code=500,
            detail="Unable to process your question right now. Please try again later.",
        ) from None


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    replace: bool = Query(True, description="Replace entire index (True) or add (False)"),
):
    require_api_ready()

    tmp_path: str | None = None
    try:
        content = await file.read()
        validate_pdf_upload(file, content)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info("Received PDF upload: %s (%s bytes)", file.filename, len(content))
        chunks_count, pages_count = process_pdf_and_update_index(
            tmp_path, replace=replace, filename=file.filename
        )

        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "pages": pages_count,
            "chunks": chunks_count,
            "status": "ready",
        }
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Upload rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Error processing PDF upload")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the PDF right now. Please try a smaller file.",
        ) from None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/demo/load-sample", response_model=SampleLoadResponse)
async def load_sample():
    """Load the bundled sample PDF into the FAISS index (demo / Try sample document)."""
    require_api_ready()

    try:
        chunks_count, pages_count, file_name = load_sample_document()
        return SampleLoadResponse(
            status="sample_loaded",
            document=SampleDocumentInfo(
                file_name=file_name,
                pages=pages_count,
                chunks=chunks_count,
            ),
            index_ready=state.is_index_ready(),
        )
    except FileNotFoundError as exc:
        logger.warning("Sample document missing: %s", exc)
        raise HTTPException(
            status_code=404,
            detail="Sample document is not available on this server.",
        ) from exc
    except ValueError as exc:
        logger.warning("Sample document rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Error loading sample document")
        raise HTTPException(
            status_code=500,
            detail="Unable to load the sample document right now. Please try again later.",
        ) from None


@router.get("/")
def home():
    payload = build_status_payload()
    return {
        "status": "DocMind API online",
        "endpoints": [
            "/ask (POST) - Ask questions about documents",
            "/upload (POST) - Upload PDF files to process",
            "/demo/load-sample (POST) - Load bundled demo PDF",
            "/health (GET) - Health and limits",
            "/status (GET) - Index and document state",
            "/clear (DELETE) - Clear/reset the document index",
        ],
        "health": payload["status"],
        "index_ready": payload["index_ready"],
    }


@router.get("/health")
def health():
    return build_health_payload()


@router.get("/status")
def status():
    return build_status_payload()


@router.delete("/clear")
def clear_index():
    try:
        clear_vector_index()
        return {
            "status": "cleared",
            "index_ready": False,
            "message": "Index cleared successfully.",
        }
    except Exception:
        logger.exception("Error clearing index")
        raise HTTPException(
            status_code=500,
            detail="Unable to clear the index right now. Please try again later.",
        ) from None
