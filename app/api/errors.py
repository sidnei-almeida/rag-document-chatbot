"""Consistent API error responses for the frontend contract."""

from fastapi import HTTPException


def api_error(status_code: int, message: str) -> HTTPException:
    """Return HTTPException with body shape: {"error": "..."}."""
    return HTTPException(status_code=status_code, detail={"error": message})
