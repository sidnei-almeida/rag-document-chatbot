"""DocMind API route handlers."""

import logging
import time

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from groq import RateLimitError

from app.api.rag_guard import require_api_ready, require_llm_ready, require_workspace_ready
from app.api.errors import api_error
from app.api.validators import read_and_validate_upload_files, validate_question
from app.core.config import settings
from app.core.state import state
from app.prompts.rag import create_general_prompt, create_rag_prompt, is_general_question
from app.schemas.ask import AskResponse, QuestionRequest
from app.schemas.demo import SampleDocumentInfo, SampleLoadResponse
from app.schemas.workspace import WorkspaceUploadResponse
from app.services.retrieval import (
    build_ask_response,
    compute_confidence,
    filter_useful_documents_with_scores,
    format_context_from_documents,
    format_sources_for_workspace,
    retrieve_documents_for_workspace,
)
from app.services.sample_loader import load_sample_workspace
from app.services.status import build_health_payload, build_status_payload
from app.services.workspace.ingestion import create_workspace_from_uploads
from app.services.workspace.service import (
    delete_all_workspaces,
    delete_workspace,
    resolve_workspace_id,
)
logger = logging.getLogger("docmind")
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_document(req: QuestionRequest):
    started = time.perf_counter()
    try:
        question = validate_question(req.question)

        general = is_general_question(question)

        if general:
            require_llm_ready()
            from app.services.workspace.registry import get_default_workspace_id

            workspace_id = (
                req.workspace_id
                or req.conversation_id
                or get_default_workspace_id()
                or "none"
            )
            prompt = create_general_prompt(question)
            response = state.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return build_ask_response(
                workspace_id=workspace_id,
                answer=answer,
                sources=[],
                confidence_label="medium",
                confidence_reason="General conversational question; no document retrieval performed.",
                retrieval_used=False,
                latency_ms=latency_ms,
            )

        try:
            workspace_id = resolve_workspace_id(
                workspace_id=req.workspace_id,
                conversation_id=req.conversation_id,
            )
        except ValueError as exc:
            raise api_error(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise api_error(404, str(exc)) from exc

        require_workspace_ready(workspace_id)
        docs, scores = retrieve_documents_for_workspace(workspace_id, question)
        useful_docs, aligned_scores = filter_useful_documents_with_scores(docs, scores)

        if not useful_docs:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return build_ask_response(
                workspace_id=workspace_id,
                answer=settings.NO_EVIDENCE_MESSAGE,
                sources=[],
                confidence_label="low",
                confidence_reason="No reliable document context was found.",
                retrieval_used=True,
                latency_ms=latency_ms,
            )

        structured_sources = format_sources_for_workspace(
            workspace_id, useful_docs, aligned_scores
        )
        confidence_label, confidence_reason = compute_confidence(
            len(structured_sources), has_context=True, scores=aligned_scores
        )
        context = format_context_from_documents(useful_docs)
        prompt = create_rag_prompt(question, context)
        response = state.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        latency_ms = int((time.perf_counter() - started) * 1000)

        return build_ask_response(
            workspace_id=workspace_id,
            answer=answer,
            sources=structured_sources,
            confidence_label=confidence_label,
            confidence_reason=confidence_reason,
            retrieval_used=True,
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except RateLimitError as exc:
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


@router.post("/upload", response_model=WorkspaceUploadResponse)
async def upload_pdf_legacy(file: UploadFile = File(...)):
    """
    Legacy upload (field `file`). Prefer POST /workspaces/upload with `files`.

    Internally creates a new isolated workspace.
    """
    require_api_ready()
    try:
        uploads = await read_and_validate_upload_files([file])
        result = create_workspace_from_uploads(uploads, source="upload")
        return WorkspaceUploadResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise api_error(400, str(exc)) from exc
    except Exception:
        logger.exception("Legacy upload failed")
        raise api_error(500, "Unable to process the PDF.") from None


@router.post("/upload/batch", response_model=WorkspaceUploadResponse, deprecated=True)
async def upload_batch_legacy(
    files: list[UploadFile] = File(..., description="Use POST /workspaces/upload instead"),
):
    """Deprecated — use POST /workspaces/upload with multiple `files`."""
    require_api_ready()
    try:
        uploads = await read_and_validate_upload_files(files)
        result = create_workspace_from_uploads(uploads, source="upload")
        return WorkspaceUploadResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise api_error(400, str(exc)) from exc
    except Exception:
        logger.exception("Legacy batch upload failed")
        raise api_error(500, "Unable to process PDFs.") from None


@router.post("/demo/load-sample", response_model=SampleLoadResponse)
async def load_sample():
    require_api_ready()
    try:
        result = load_sample_workspace()
        doc = result["documents"][0]
        return SampleLoadResponse(
            status="sample_loaded",
            workspace_id=result["workspace_id"],
            document=SampleDocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                pages=doc["pages"],
                chunks=doc["chunks"],
            ),
            index_ready=result["index_ready"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Error loading sample workspace")
        raise HTTPException(status_code=500, detail="Unable to load sample.") from None


@router.get("/")
def home():
    payload = build_status_payload()
    return {
        "status": "DocMind API online",
        "endpoints": [
            "/workspaces/upload (POST) - Upload 1+ PDFs → new workspace",
            "/workspaces (GET) - List workspaces",
            "/workspaces/{workspace_id} (GET|DELETE)",
            "/ask (POST) - Ask with workspace_id",
            "/upload (POST) - Legacy single PDF → new workspace",
            "/demo/load-sample (POST) - Demo workspace from sample PDF",
            "/health (GET)",
            "/status (GET)",
            "/clear (DELETE) - Clear all or one workspace",
        ],
        "default_workspace_id": payload.get("default_workspace_id"),
        "index_ready": payload.get("index_ready"),
    }


@router.get("/health")
def health():
    return build_health_payload()


@router.get("/status")
def status():
    return build_status_payload()


@router.delete("/clear")
def clear_index(workspace_id: str | None = Query(None)):
    try:
        if workspace_id:
            delete_workspace(workspace_id)
            return {
                "status": "deleted",
                "workspace_id": workspace_id,
                "message": f"Workspace {workspace_id} removed.",
            }
        count = delete_all_workspaces()
        return {
            "status": "cleared",
            "workspaces_removed": count,
            "message": "All workspaces removed.",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception("Error clearing workspace(s)")
        raise HTTPException(status_code=500, detail="Unable to clear.") from None
