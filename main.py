"""Backward-compatible entrypoint (uvicorn main:app or imports from legacy code)."""

from app.main import app

__all__ = ["app"]
