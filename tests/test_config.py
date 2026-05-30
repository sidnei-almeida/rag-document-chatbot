"""Configuration helper tests."""

from app.core.config import DEFAULT_CORS_ORIGINS, get_cors_origins, settings


def test_settings_workspace_limits_defaults():
    assert settings.MAX_FILE_SIZE_MB == 20
    assert settings.MAX_FILES_PER_WORKSPACE == 5
    assert settings.MAX_PAGES_PER_FILE == 40
    assert settings.MAX_TOTAL_PAGES == 100
    assert settings.MAX_QUESTION_LENGTH == 1000
    assert settings.RETRIEVAL_K == 6


def test_get_cors_origins_includes_local_frontends():
    origins = get_cors_origins()
    assert "http://localhost:3000" in origins
    assert "http://localhost:5173" in origins


def test_default_cors_origins_tuple():
    assert "http://localhost:3000" in DEFAULT_CORS_ORIGINS


def test_retrieval_dict_contains_mmr_fields():
    retrieval = settings.retrieval_dict()
    assert retrieval["type"] == "mmr"
    assert retrieval["k"] == settings.RETRIEVAL_K
