"""Tests for NLP pipeline module."""

import pytest

from latincyreaders.nlp.pipeline import (
    AnnotationLevel,
    create_pipeline,
    get_nlp,
    load_model,
)


class TestAnnotationLevel:
    """Tests for AnnotationLevel enum."""

    def test_none_level(self):
        """NONE level creates no pipeline."""
        nlp = create_pipeline(AnnotationLevel.NONE)
        assert nlp is None

    def test_tokenize_level(self):
        """TOKENIZE level creates minimal pipeline."""
        nlp = create_pipeline(AnnotationLevel.TOKENIZE)
        assert nlp is not None
        assert "sentencizer" in nlp.pipe_names

    def test_basic_level(self):
        """BASIC level loads model without NER/parser."""
        nlp = create_pipeline(AnnotationLevel.BASIC)
        assert nlp is not None
        # NER and parser should be disabled
        assert "ner" not in nlp.pipe_names

    def test_full_level(self):
        """FULL level loads model with all components."""
        nlp = create_pipeline(AnnotationLevel.FULL)
        assert nlp is not None
        # Full model has all components
        assert nlp.max_length == 2_500_000


class TestGetNlp:
    """Tests for get_nlp function."""

    def test_get_nlp_basic(self):
        """get_nlp returns pipeline for BASIC level."""
        nlp = get_nlp(AnnotationLevel.BASIC)
        assert nlp is not None

    def test_get_nlp_none(self):
        """get_nlp returns None for NONE level."""
        nlp = get_nlp(AnnotationLevel.NONE)
        assert nlp is None


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_model_default(self):
        """load_model loads default Latin model."""
        nlp = load_model()
        assert nlp is not None
        assert nlp.max_length == 2_500_000

    def test_load_model_cached(self):
        """load_model returns cached model on repeat calls."""
        nlp1 = load_model()
        nlp2 = load_model()
        assert nlp1 is nlp2  # Same object from cache
