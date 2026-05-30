"""Tests for general-question detection (word boundaries)."""

from app.prompts.rag import is_general_question


def test_document_question_with_this_is_not_general():
    assert is_general_question("What is this document about?") is False


def test_hello_is_general():
    assert is_general_question("hello") is True


def test_ola_is_general():
    assert is_general_question("olá") is True
