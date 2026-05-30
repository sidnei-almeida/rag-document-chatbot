"""Configuration helper tests."""

from app.core.config import get_cors_origins, settings


def test_settings_demo_limits_defaults():
    assert settings.MAX_FILE_SIZE_MB == 8
    assert settings.MAX_PAGES == 40
    assert settings.MAX_QUESTION_LENGTH == 1000
    assert settings.RETRIEVAL_K == 6


def test_get_cors_origins_returns_non_empty_list():
    origins = get_cors_origins()
    assert isinstance(origins, list)
    assert len(origins) >= 1


def test_retrieval_dict_contains_mmr_fields():
    retrieval = settings.retrieval_dict()
    assert retrieval["type"] == "mmr"
    assert retrieval["k"] == settings.RETRIEVAL_K
