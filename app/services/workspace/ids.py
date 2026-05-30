"""Workspace and document id generators."""

import secrets
from datetime import datetime, timezone


def new_workspace_id() -> str:
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = secrets.token_hex(3)
    return f"ws_{date_part}_{suffix}"


def new_document_id() -> str:
    return f"doc_{secrets.token_hex(4)}"


def format_chunk_id(sequence: int) -> str:
    return f"chunk_{sequence:03d}"
