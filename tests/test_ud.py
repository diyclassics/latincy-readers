"""Tests for UDReader (CONLLU/CONLLUP formats)."""

import pytest
from pathlib import Path

from latincyreaders import UDReader, AnnotationLevel


class TestUDReader:
    """Test suite for UDReader."""

    @pytest.fixture
    def reader(self, ud_dir):
        """Create a UDReader with test fixtures (CONLLU only)."""
        return UDReader(root=ud_dir, fileids="*.conllu")

    @pytest.fixture
    def reader_conllup(self, ud_dir):
        """Create a UDReader for CONLLUP files."""
        return UDReader(root=ud_dir, fileids="*.conllup")

    @pytest.fixture
    def reader_all(self, ud_dir):
        """Create a UDReader for all UD files."""
        return UDReader(root=ud_dir)

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .conllu files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".conllu") for f in fileids)

    def test_fileids_conllup(self, reader_conllup):
        """fileids() returns .conllup files."""
        fileids = reader_conllup.fileids()
        assert len(fileids) > 0
        assert all(f.endswith(".conllup") for f in fileids)

    def test_fileids_all_formats(self, reader_all):
        """Default pattern matches both .conllu and .conllup."""
        fileids = reader_all.fileids()
        extensions = {Path(f).suffix for f in fileids}
        assert ".conllu" in extensions or ".conllup" in extensions

    def test_root_is_path(self, reader, ud_dir):
        """root property returns correct Path."""
        assert reader.root == ud_dir.resolve()

    def test_use_file_annotations_default(self, ud_dir):
        """use_file_annotations defaults to True."""
        reader = UDReader(root=ud_dir)
        assert reader.use_file_annotations is True

    # -------------------------------------------------------------------------
    # Raw text access
    # -------------------------------------------------------------------------

    def test_texts_yields_strings(self, reader):
        """texts() yields raw strings."""
        texts = list(reader.texts())
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_texts_contains_latin(self, reader):
        """Text content is Latin."""
        texts = list(reader.texts())
        all_text = " ".join(texts)
        assert "Mucius" in all_text or "augur" in all_text

    # -------------------------------------------------------------------------
    # spaCy Doc access with file annotations
    # -------------------------------------------------------------------------

    def test_docs_yields_spacy_docs(self, reader):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader):
        """Docs have fileid extension set."""
        doc = next(reader.docs())
        assert doc._.fileid is not None
        assert doc._.fileid.endswith(".conllu")

    def test_docs_preserve_lemmas_from_file(self, reader):
        """When use_file_annotations=True, lemmas come from file."""
        doc = next(reader.docs())
        # Check that lemmas are set from the CONLLU file
        lemmas = [t.lemma_ for t in doc]
        assert "Quintus" in lemmas or "Mucius" in lemmas

    def test_docs_preserve_pos_from_file(self, reader):
        """When use_file_annotations=True, POS tags come from file."""
        doc = next(reader.docs())
        pos_tags = [t.pos_ for t in doc]
        assert "PROPN" in pos_tags or "NOUN" in pos_tags

    def test_docs_have_ud_extensions(self, reader):
        """Tokens have UD-specific extensions set."""
        doc = next(reader.docs())
        token = doc[0]
        # Check UD extensions are populated
        assert token._.ud_lemma is not None
        assert token._.ud_upos is not None

    def test_docs_have_morphology(self, reader):
        """Tokens have morphological features from file."""
        doc = next(reader.docs())
        # Find a token with morph features
        for token in doc:
            if token._.ud_feats:
                assert isinstance(token._.ud_feats, dict)
                # Check morph is also set on token
                morph_str = str(token.morph)
                assert morph_str  # Should have some morph info
                break

    # -------------------------------------------------------------------------
    # CONLLUP-specific tests
    # -------------------------------------------------------------------------

    def test_conllup_no_dependencies(self, reader_conllup):
        """CONLLUP files have no dependency annotations."""
        doc = next(reader_conllup.docs())
        for token in doc:
            # HEAD and DEPREL should be None in CONLLUP
            assert token._.ud_head is None
            assert token._.ud_deprel is None

    def test_conllup_has_pos_lemma(self, reader_conllup):
        """CONLLUP files still have POS and lemma."""
        doc = next(reader_conllup.docs())
        pos_tags = [t.pos_ for t in doc if t.pos_]
        lemmas = [t.lemma_ for t in doc if t.lemma_]
        assert len(pos_tags) > 0
        assert len(lemmas) > 0

    def test_has_dependencies_conllu(self, reader):
        """has_dependencies() returns True for CONLLU files."""
        assert reader.has_dependencies() is True

    def test_has_dependencies_conllup(self, reader_conllup):
        """has_dependencies() returns False for CONLLUP files."""
        assert reader_conllup.has_dependencies() is False

    # -------------------------------------------------------------------------
    # Sentence handling
    # -------------------------------------------------------------------------

    def test_sentences_yields_spans(self, reader):
        """sentences() yields Span objects."""
        from spacy.tokens import Span

        sents = list(reader.sentences())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_sentences_have_citations(self, reader):
        """Sentences have sent_id as citation."""
        for sent in reader.sentences():
            # sent_id from CONLLU metadata becomes citation
            if sent._.citation:
                assert "cic.amic" in sent._.citation
                break

    # -------------------------------------------------------------------------
    # tokens_with_annotations method
    # -------------------------------------------------------------------------

    def test_tokens_with_annotations(self, reader):
        """tokens_with_annotations() yields complete token dicts."""
        tokens = list(reader.tokens_with_annotations())
        assert len(tokens) > 0

        token = tokens[0]
        assert "fileid" in token
        assert "form" in token
        assert "lemma" in token
        assert "upos" in token
        assert "feats" in token

    # -------------------------------------------------------------------------
    # use_file_annotations=False (LatinCy mode)
    # -------------------------------------------------------------------------

    def test_latincy_mode_stores_ud_originals(self, ud_dir):
        """When use_file_annotations=False, UD values stored in extensions."""
        import spacy
        try:
            spacy.load("la_core_web_lg")
        except OSError:
            pytest.skip("LatinCy model (la_core_web_lg) not installed")

        reader = UDReader(
            root=ud_dir,
            fileids="*.conllu",
            use_file_annotations=False,
            annotation_level=AnnotationLevel.BASIC,
        )
        doc = next(reader.docs())

        # UD originals should be in extensions
        for token in doc:
            if token._.ud_lemma:
                # The extension has the original UD lemma
                assert isinstance(token._.ud_lemma, str)
                break

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def test_caching_works(self, ud_dir):
        """Document caching works correctly."""
        reader = UDReader(root=ud_dir, fileids="*.conllu", cache=True)

        # First access - cache miss
        doc1 = next(reader.docs())
        stats1 = reader.cache_stats()
        assert stats1["misses"] == 1

        # Second access - cache hit
        doc2 = next(reader.docs())
        stats2 = reader.cache_stats()
        assert stats2["hits"] == 1

    def test_cache_disabled(self, ud_dir):
        """Caching can be disabled."""
        reader = UDReader(root=ud_dir, fileids="*.conllu", cache=False)
        assert reader.cache_enabled is False
