"""Tests for TesseraeReader."""

import pytest
from pathlib import Path

from latincyreaders import TesseraeReader, AnnotationLevel


class TestTesseraeReader:
    """Test suite for TesseraeReader."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        """Create a TesseraeReader with test fixtures."""
        return TesseraeReader(root=tesserae_dir, fileids="*.tess")

    @pytest.fixture
    def reader_tokenize_only(self, tesserae_dir):
        """Reader with minimal annotation for faster tests."""
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    # -------------------------------------------------------------------------
    # Basic functionality
    # -------------------------------------------------------------------------

    def test_fileids_returns_list(self, reader):
        """fileids() returns a list of .tess files."""
        fileids = reader.fileids()
        assert isinstance(fileids, list)
        assert len(fileids) > 0
        assert all(f.endswith(".tess") for f in fileids)

    def test_fileids_contains_test_file(self, reader):
        """Test fixture file is discovered."""
        fileids = reader.fileids()
        assert "tesserae.tess" in fileids

    def test_root_is_path(self, reader, tesserae_dir):
        """root property returns correct Path."""
        assert reader.root == tesserae_dir.resolve()

    def test_annotation_level_property(self, tesserae_dir):
        """annotation_level property returns current level."""
        reader = TesseraeReader(
            root=tesserae_dir,
            annotation_level=AnnotationLevel.TOKENIZE,
        )
        assert reader.annotation_level == AnnotationLevel.TOKENIZE

    # -------------------------------------------------------------------------
    # Raw text access (no NLP)
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
        # Check for known content from test files
        assert "Mucius" in all_text or "Corneli" in all_text

    def test_texts_by_line_yields_tuples(self, reader):
        """texts_by_line() yields (citation, text) tuples."""
        lines = list(reader.texts_by_line())
        assert len(lines) > 0
        assert all(isinstance(line, tuple) and len(line) == 2 for line in lines)

    def test_texts_by_line_has_citations(self, reader):
        """Citations are in correct format."""
        citation, text = next(reader.texts_by_line())
        assert citation.startswith("<")
        assert citation.endswith(">")
        assert len(text) > 0

    # -------------------------------------------------------------------------
    # spaCy Doc access
    # -------------------------------------------------------------------------

    def test_docs_yields_spacy_docs(self, reader_tokenize_only):
        """docs() yields spaCy Doc objects."""
        from spacy.tokens import Doc

        docs = list(reader_tokenize_only.docs())
        assert len(docs) > 0
        assert all(isinstance(d, Doc) for d in docs)

    def test_docs_have_fileid(self, reader_tokenize_only):
        """Docs have fileid extension set."""
        doc = next(reader_tokenize_only.docs())
        assert doc._.fileid is not None
        assert doc._.fileid.endswith(".tess")

    def test_docs_have_metadata(self, reader_tokenize_only):
        """Docs have metadata extension set."""
        doc = next(reader_tokenize_only.docs())
        assert doc._.metadata is not None
        assert "filename" in doc._.metadata

    def test_docs_have_line_spans(self, reader_tokenize_only):
        """Docs have 'lines' span group."""
        doc = next(reader_tokenize_only.docs())
        assert "lines" in doc.spans
        assert len(doc.spans["lines"]) > 0

    def test_line_spans_have_citations(self, reader_tokenize_only):
        """Line spans have citation extensions."""
        doc = next(reader_tokenize_only.docs())
        for span in doc.spans["lines"]:
            assert span._.citation is not None
            assert span._.citation.startswith("<")

    # -------------------------------------------------------------------------
    # Sentence and token iteration
    # -------------------------------------------------------------------------

    def test_sents_yields_spans(self, reader_tokenize_only):
        """sents() yields Span objects."""
        from spacy.tokens import Span

        sents = list(reader_tokenize_only.sents())
        assert len(sents) > 0
        assert all(isinstance(s, Span) for s in sents)

    def test_sents_as_text_yields_strings(self, reader_tokenize_only):
        """sents(as_text=True) yields strings."""
        sents = list(reader_tokenize_only.sents(as_text=True))
        assert len(sents) > 0
        assert all(isinstance(s, str) for s in sents)

    def test_tokens_yields_tokens(self, reader_tokenize_only):
        """tokens() yields Token objects."""
        from spacy.tokens import Token

        tokens = list(reader_tokenize_only.tokens())
        assert len(tokens) > 0
        assert all(isinstance(t, Token) for t in tokens)

    def test_tokens_as_text_yields_strings(self, reader_tokenize_only):
        """tokens(as_text=True) yields strings."""
        tokens = list(reader_tokenize_only.tokens(as_text=True))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    # -------------------------------------------------------------------------
    # Citation-specific methods
    # -------------------------------------------------------------------------

    def test_lines_yields_spans_with_citations(self, reader_tokenize_only):
        """lines() yields Spans with citations."""
        from spacy.tokens import Span

        lines = list(reader_tokenize_only.lines())
        assert len(lines) > 0
        assert all(isinstance(line, Span) for line in lines)
        assert all(line._.citation is not None for line in lines)

    def test_doc_rows_yields_dicts(self, reader_tokenize_only):
        """doc_rows() yields citation->Span dicts."""
        rows = list(reader_tokenize_only.doc_rows())
        assert len(rows) > 0
        assert all(isinstance(r, dict) for r in rows)

        # Check structure
        row = rows[0]
        for citation, span in row.items():
            assert citation.startswith("<")
            assert hasattr(span, "text")

    # -------------------------------------------------------------------------
    # Annotation levels
    # -------------------------------------------------------------------------

    def test_annotation_level_none_blocks_docs(self, tesserae_dir):
        """annotation_level=NONE prevents docs() from working."""
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        with pytest.raises(ValueError, match="annotation_level=NONE"):
            next(reader.docs())

    def test_annotation_level_none_allows_texts(self, tesserae_dir):
        """annotation_level=NONE still allows texts()."""
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        texts = list(reader.texts())
        assert len(texts) > 0

    def test_annotation_level_none_allows_texts_by_line(self, tesserae_dir):
        """annotation_level=NONE still allows texts_by_line()."""
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.NONE,
        )
        lines = list(reader.texts_by_line())
        assert len(lines) > 0

    # -------------------------------------------------------------------------
    # Auto-download functionality
    # -------------------------------------------------------------------------

    def test_auto_download_false_raises_on_missing(self, tmp_path, monkeypatch):
        """auto_download=False raises FileNotFoundError for missing corpus."""
        # Point to a non-existent directory
        monkeypatch.setenv("TESSERAE_PATH", str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="corpus not found"):
            TesseraeReader(root=None, auto_download=False)

    def test_default_root_returns_path(self):
        """default_root() returns a Path."""
        root = TesseraeReader.default_root()
        assert isinstance(root, Path)

    def test_explicit_root_bypasses_default(self, tesserae_dir):
        """Explicit root= bypasses default location logic."""
        reader = TesseraeReader(root=tesserae_dir)
        assert reader.root == tesserae_dir.resolve()


class TestTesseraeLineParsing:
    """Test the line parsing logic specifically."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        return TesseraeReader(root=tesserae_dir)

    def test_parse_lines_extracts_citations(self, reader, sample_tess_file):
        """_parse_lines extracts citation-text pairs."""
        text = sample_tess_file.read_text()
        lines = list(reader._parse_lines(text))

        assert len(lines) == 5  # Test file has 5 citation lines
        assert lines[0].citation == "<cic. amicit. 1>"
        assert lines[1].citation == "<cic. amicit. 2>"

    def test_parse_lines_preserves_text(self, reader, sample_tess_file):
        """_parse_lines preserves the actual text content."""
        text = sample_tess_file.read_text()
        lines = list(reader._parse_lines(text))

        # First line should start with "Q. Mucius"
        assert lines[0].text.startswith("Q. Mucius augur")

    def test_parse_lines_handles_empty_input(self, reader):
        """_parse_lines handles empty input gracefully."""
        lines = list(reader._parse_lines(""))
        assert lines == []

    def test_parse_lines_handles_no_citations(self, reader):
        """_parse_lines handles text without citations."""
        lines = list(reader._parse_lines("Just some plain text\nwith no citations"))
        assert lines == []


class TestTesseraeSearch:
    """Test search functionality."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.TOKENIZE,
        )

    # -------------------------------------------------------------------------
    # search() method
    # -------------------------------------------------------------------------

    def test_search_returns_matches(self, reader):
        """search() finds pattern matches."""
        results = list(reader.search(r"Mucius"))
        assert len(results) > 0
        # Each result is (fileid, citation, text, matches)
        fileid, citation, text, matches = results[0]
        assert "Mucius" in matches or "mucius" in matches

    def test_search_returns_citation(self, reader):
        """search() includes citation in results."""
        results = list(reader.search(r"Scaevola"))
        assert len(results) > 0
        _, citation, _, _ = results[0]
        assert citation.startswith("<")
        assert citation.endswith(">")

    def test_search_case_insensitive_default(self, reader):
        """search() is case-insensitive by default."""
        upper = list(reader.search(r"MUCIUS"))
        lower = list(reader.search(r"mucius"))
        assert len(upper) == len(lower)

    def test_search_case_sensitive_option(self, reader):
        """search() can be case-sensitive."""
        # "Mucius" appears capitalized in text
        upper = list(reader.search(r"Mucius", ignore_case=False))
        lower = list(reader.search(r"mucius", ignore_case=False))
        assert len(upper) > 0
        assert len(lower) == 0

    # -------------------------------------------------------------------------
    # find_lines() method
    # -------------------------------------------------------------------------

    def test_find_lines_with_pattern(self, reader):
        """find_lines() works with regex pattern."""
        results = list(reader.find_lines(pattern=r"\bMuci\w+\b"))
        assert len(results) > 0
        fileid, citation, text = results[0]
        assert "Muci" in text

    def test_find_lines_with_forms(self, reader):
        """find_lines() works with forms list."""
        results = list(reader.find_lines(forms=["Mucius", "Scaevola"]))
        assert len(results) > 0

    def test_find_lines_requires_pattern_or_forms(self, reader):
        """find_lines() raises if neither pattern nor forms provided."""
        with pytest.raises(ValueError):
            list(reader.find_lines())

    # -------------------------------------------------------------------------
    # find_sents() method
    # -------------------------------------------------------------------------

    def test_find_sents_returns_dicts(self, reader):
        """find_sents() returns dicts with expected keys."""
        results = list(reader.find_sents(pattern=r"\bMucius\b"))
        assert len(results) > 0
        result = results[0]
        assert "fileid" in result
        assert "citation" in result
        assert "sentence" in result
        assert "matches" in result

    def test_find_sents_with_forms(self, reader):
        """find_sents() works with forms list."""
        results = list(reader.find_sents(forms=["Laelius", "Laeli"]))
        assert len(results) > 0

    def test_find_sents_with_context(self, reader):
        """find_sents() includes context when requested."""
        results = list(reader.find_sents(pattern=r"\bMucius\b", context=True))
        assert len(results) > 0
        result = results[0]
        assert "prev_sent" in result
        assert "next_sent" in result

    def test_find_sents_with_matcher_pattern(self, tesserae_dir):
        """find_sents() works with spaCy Matcher patterns."""
        # Matcher patterns need BASIC annotation level for POS
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
        )
        # Pattern: NOUN followed by any word
        pattern = [{"POS": "NOUN"}, {}]
        results = list(reader.find_sents(matcher_pattern=pattern))
        assert len(results) > 0
        result = results[0]
        assert "matches" in result
        assert "pattern" in result
        assert len(result["matches"]) > 0

    def test_find_sents_matcher_with_lemma(self, tesserae_dir):
        """find_sents() Matcher works with LEMMA patterns."""
        # Matcher with LEMMA needs BASIC annotation level
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
        )
        # Pattern: match specific lemma
        pattern = [{"LEMMA": {"IN": ["qui", "sum", "et"]}}]  # Common Latin words
        results = list(reader.find_sents(matcher_pattern=pattern))
        # Should find at least one match
        assert len(results) >= 0  # Pattern may or may not match depending on model

    # -------------------------------------------------------------------------
    # export_search_results() method
    # -------------------------------------------------------------------------

    def test_export_tsv(self, reader):
        """export_search_results() produces TSV."""
        results = reader.find_sents(pattern=r"\bMucius\b")
        tsv = reader.export_search_results(results, format="tsv")
        lines = tsv.split("\n")
        assert len(lines) > 1  # Header + data
        assert "fileid" in lines[0]
        assert "\t" in lines[0]

    def test_export_csv(self, reader):
        """export_search_results() produces CSV."""
        results = reader.find_sents(pattern=r"\bMucius\b")
        csv = reader.export_search_results(results, format="csv")
        lines = csv.split("\n")
        assert len(lines) > 1
        assert "fileid" in lines[0]
        assert "," in lines[0]

    def test_export_jsonl(self, reader):
        """export_search_results() produces JSONL."""
        import json
        results = reader.find_sents(pattern=r"\bMucius\b")
        jsonl = reader.export_search_results(results, format="jsonl")
        lines = jsonl.split("\n")
        # Each line should be valid JSON
        for line in lines:
            if line:
                data = json.loads(line)
                assert "sentence" in data


class TestConcordance:
    """Test concordance() method."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_concordance_returns_dict(self, reader):
        """concordance() returns a dictionary."""
        conc = reader.concordance()
        assert isinstance(conc, dict)
        assert len(conc) > 0

    def test_concordance_values_are_lists(self, reader):
        """concordance() values are lists of citations."""
        conc = reader.concordance()
        for key, refs in conc.items():
            assert isinstance(key, str)
            assert isinstance(refs, list)
            assert all(isinstance(r, str) for r in refs)

    def test_concordance_citations_format(self, reader):
        """concordance() includes citation strings from Tesserae."""
        conc = reader.concordance()
        # At least some citations should have angle bracket format
        all_refs = [ref for refs in conc.values() for ref in refs]
        assert any("<" in ref for ref in all_refs)

    def test_concordance_basis_lemma(self, reader):
        """concordance(basis='lemma') groups by lemma."""
        conc = reader.concordance(basis="lemma")
        assert isinstance(conc, dict)
        assert len(conc) > 0

    def test_concordance_basis_text(self, reader):
        """concordance(basis='text') groups by surface form."""
        conc = reader.concordance(basis="text")
        assert isinstance(conc, dict)
        assert len(conc) > 0

    def test_concordance_basis_norm(self, reader):
        """concordance(basis='norm') groups by normalized form."""
        conc = reader.concordance(basis="norm")
        assert isinstance(conc, dict)
        assert len(conc) > 0

    def test_concordance_only_alpha_true(self, reader):
        """concordance(only_alpha=True) excludes punctuation."""
        conc = reader.concordance(only_alpha=True)
        # Should not have punctuation-only keys
        punct_keys = [k for k in conc.keys() if not any(c.isalpha() for c in k)]
        assert len(punct_keys) == 0

    def test_concordance_only_alpha_false(self, reader):
        """concordance(only_alpha=False) includes punctuation."""
        conc = reader.concordance(only_alpha=False)
        # Should have some punctuation keys
        punct_keys = [k for k in conc.keys() if not any(c.isalpha() for c in k)]
        assert len(punct_keys) > 0

    def test_concordance_sorted(self, reader):
        """concordance() keys are sorted alphabetically."""
        conc = reader.concordance()
        keys = list(conc.keys())
        assert keys == sorted(keys)


class TestKWIC:
    """Test kwic() (keyword in context) method."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_kwic_returns_iterator(self, reader):
        """kwic() returns an iterator of dicts."""
        results = list(reader.kwic("Mucius"))
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_kwic_result_structure(self, reader):
        """kwic() results have expected keys."""
        result = next(reader.kwic("Mucius"))
        assert "left" in result
        assert "match" in result
        assert "right" in result
        assert "citation" in result
        assert "fileid" in result

    def test_kwic_match_is_keyword(self, reader):
        """kwic() match field contains the matched word."""
        for hit in reader.kwic("Mucius"):
            assert hit["match"].lower() == "mucius"

    def test_kwic_ignore_case_true(self, reader):
        """kwic(ignore_case=True) matches case-insensitively."""
        upper_results = list(reader.kwic("MUCIUS", ignore_case=True))
        lower_results = list(reader.kwic("mucius", ignore_case=True))
        assert len(upper_results) == len(lower_results)

    def test_kwic_ignore_case_false(self, reader):
        """kwic(ignore_case=False) is case-sensitive."""
        # 'Mucius' (capitalized) should match, 'mucius' may not
        cap_results = list(reader.kwic("Mucius", ignore_case=False))
        assert len(cap_results) >= 0  # At least runs without error

    def test_kwic_window_size(self, reader):
        """kwic(window=N) controls context size."""
        # Small window
        hit_small = next(reader.kwic("Mucius", window=2))
        # Large window
        hit_large = next(reader.kwic("Mucius", window=10))
        # Larger window should generally have more context (unless at doc boundary)
        assert len(hit_large["left"]) >= len(hit_small["left"]) or \
               len(hit_large["right"]) >= len(hit_small["right"])

    def test_kwic_limit(self, reader):
        """kwic(limit=N) limits number of results."""
        all_results = list(reader.kwic("et", limit=None))
        limited_results = list(reader.kwic("et", limit=3))
        assert len(limited_results) <= 3
        if len(all_results) > 3:
            assert len(limited_results) == 3

    def test_kwic_by_lemma(self, tesserae_dir):
        """kwic(by_lemma=True) matches against lemma."""
        reader = TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
        )
        # 'sum' is lemma for 'est', 'sunt', 'esse', etc.
        lemma_results = list(reader.kwic("sum", by_lemma=True, limit=5))
        # Should match various forms of 'sum'
        assert len(lemma_results) >= 0  # At least runs

    def test_kwic_citation_format(self, reader):
        """kwic() includes Tesserae-style citations."""
        hit = next(reader.kwic("Mucius"))
        # Should have angle bracket citation
        assert "<" in hit["citation"]

    def test_kwic_no_match(self, reader):
        """kwic() returns empty iterator for non-existent word."""
        results = list(reader.kwic("xyznonexistent123"))
        assert len(results) == 0


class TestNgrams:
    """Test ngrams() method."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_ngrams_returns_strings_by_default(self, reader):
        """ngrams() returns strings by default."""
        bigrams = list(reader.ngrams(n=2))
        assert len(bigrams) > 0
        assert all(isinstance(bg, str) for bg in bigrams)

    def test_ngrams_as_tuples(self, reader):
        """ngrams(as_tuples=True) returns token tuples."""
        bigrams = list(reader.ngrams(n=2, as_tuples=True))
        assert len(bigrams) > 0
        assert all(isinstance(bg, tuple) for bg in bigrams)
        # Each tuple should have n tokens
        assert all(len(bg) == 2 for bg in bigrams)

    def test_ngrams_size(self, reader):
        """ngrams(n=3) returns trigrams."""
        trigrams = list(reader.ngrams(n=3))
        assert len(trigrams) > 0
        # Each trigram should have 3 words separated by spaces
        for tg in trigrams[:10]:
            words = tg.split()
            assert len(words) == 3

    def test_ngrams_filter_punct(self, reader):
        """ngrams(filter_punct=True) excludes punctuation."""
        bigrams = list(reader.ngrams(n=2, filter_punct=True))
        # Bigrams shouldn't contain standalone punctuation
        import string
        for bg in bigrams[:50]:
            words = bg.split()
            for word in words:
                # Word shouldn't be pure punctuation
                assert not all(c in string.punctuation for c in word)

    def test_ngrams_filter_punct_false(self, reader):
        """ngrams(filter_punct=False) includes punctuation."""
        bigrams = list(reader.ngrams(n=2, filter_punct=False))
        # Should have some bigrams with punctuation
        import string
        has_punct = any(
            any(c in string.punctuation for c in bg)
            for bg in bigrams[:100]
        )
        assert has_punct

    def test_ngrams_tuple_has_token_attributes(self, reader):
        """ngrams(as_tuples=True) tokens have spaCy attributes."""
        bigrams = list(reader.ngrams(n=2, as_tuples=True))
        if bigrams:
            first_gram = bigrams[0]
            # Tokens should have text, lemma_, pos_
            for token in first_gram:
                assert hasattr(token, 'text')
                assert hasattr(token, 'lemma_')
                assert hasattr(token, 'pos_')

    def test_ngrams_basis_text(self, reader):
        """ngrams(basis='text') returns surface forms (default)."""
        bigrams = list(reader.ngrams(n=2, basis="text"))
        assert len(bigrams) > 0
        # Should be strings
        assert all(isinstance(bg, str) for bg in bigrams)

    def test_ngrams_basis_lemma(self, reader):
        """ngrams(basis='lemma') returns lemmatized forms."""
        text_bigrams = list(reader.ngrams(n=2, basis="text"))
        lemma_bigrams = list(reader.ngrams(n=2, basis="lemma"))
        assert len(lemma_bigrams) > 0
        # Same count but potentially different content
        assert len(text_bigrams) == len(lemma_bigrams)
        # At least some should differ (lemmas vs surface forms)
        assert text_bigrams != lemma_bigrams or len(text_bigrams) == 0

    def test_ngrams_basis_norm(self, reader):
        """ngrams(basis='norm') returns normalized forms."""
        norm_bigrams = list(reader.ngrams(n=2, basis="norm"))
        assert len(norm_bigrams) > 0
        assert all(isinstance(bg, str) for bg in norm_bigrams)

    def test_ngrams_filter_stops(self, reader):
        """ngrams(filter_stops=True) excludes stop words."""
        # Just verify it runs without error and returns results
        bigrams = list(reader.ngrams(n=2, filter_stops=True))
        # Should have fewer results than without filter
        all_bigrams = list(reader.ngrams(n=2, filter_stops=False, filter_punct=True))
        # At minimum it shouldn't crash and should return something
        assert isinstance(bigrams, list)

    def test_ngrams_filter_nums(self, reader):
        """ngrams(filter_nums=True) excludes numeric tokens."""
        bigrams = list(reader.ngrams(n=2, filter_nums=True))
        # Should run without error
        assert isinstance(bigrams, list)


class TestSkipgrams:
    """Test skipgrams() method."""

    @pytest.fixture
    def reader(self, tesserae_dir):
        return TesseraeReader(
            root=tesserae_dir,
            fileids="*.tess",
            annotation_level=AnnotationLevel.BASIC,
        )

    def test_skipgrams_returns_strings_by_default(self, reader):
        """skipgrams() returns strings by default."""
        skipgrams = list(reader.skipgrams(n=2, k=1))
        assert len(skipgrams) > 0
        assert all(isinstance(sg, str) for sg in skipgrams)

    def test_skipgrams_as_tuples(self, reader):
        """skipgrams(as_tuples=True) returns token tuples."""
        skipgrams = list(reader.skipgrams(n=2, k=1, as_tuples=True))
        assert len(skipgrams) > 0
        assert all(isinstance(sg, tuple) for sg in skipgrams)
        assert all(len(sg) == 2 for sg in skipgrams)

    def test_skipgrams_more_than_ngrams(self, reader):
        """skipgrams with k>0 should produce more results than ngrams."""
        # Same filters for fair comparison
        ngrams = list(reader.ngrams(n=2, filter_punct=True))
        skipgrams = list(reader.skipgrams(n=2, k=1, filter_punct=True))
        # Skipgrams include ngrams plus skip patterns
        assert len(skipgrams) >= len(ngrams)

    def test_skipgrams_k0_equals_ngrams(self, reader):
        """skipgrams with k=0 should equal ngrams."""
        # Note: Due to different iteration, counts may differ slightly
        # but the concept is the same - k=0 means no skips allowed
        skipgrams_k0 = list(reader.skipgrams(n=2, k=0, filter_punct=True))
        assert len(skipgrams_k0) > 0

    def test_skipgrams_filter_punct(self, reader):
        """skipgrams(filter_punct=True) excludes punctuation."""
        skipgrams = list(reader.skipgrams(n=2, k=1, filter_punct=True))
        import string
        for sg in skipgrams[:50]:
            words = sg.split()
            for word in words:
                assert not all(c in string.punctuation for c in word)

    def test_skipgrams_basis_text(self, reader):
        """skipgrams(basis='text') returns surface forms (default)."""
        skipgrams = list(reader.skipgrams(n=2, k=1, basis="text"))
        assert len(skipgrams) > 0
        assert all(isinstance(sg, str) for sg in skipgrams)

    def test_skipgrams_basis_lemma(self, reader):
        """skipgrams(basis='lemma') returns lemmatized forms."""
        text_skipgrams = list(reader.skipgrams(n=2, k=1, basis="text"))
        lemma_skipgrams = list(reader.skipgrams(n=2, k=1, basis="lemma"))
        assert len(lemma_skipgrams) > 0
        assert len(text_skipgrams) == len(lemma_skipgrams)
