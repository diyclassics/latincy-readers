"""Base corpus reader class.

This module provides the abstract base class that all corpus readers inherit from.
It handles common functionality like file discovery, NLP pipeline management,
and the standard iteration interface.
"""

from __future__ import annotations

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING

from latincyreaders.nlp.pipeline import AnnotationLevel, get_nlp

if TYPE_CHECKING:
    from spacy import Language
    from spacy.tokens import Doc, Span, Token

# Re-export for convenience
__all__ = ["BaseCorpusReader", "AnnotationLevel"]


class BaseCorpusReader(ABC):
    """Abstract base class for all Latin corpus readers.

    To create a new reader, subclass and implement:

    Required:
        - _parse_file(path) -> yields (text, metadata) tuples

    Optional overrides:
        - _normalize_text(text) -> cleaned text
        - _default_file_pattern() -> glob pattern for files

    Example:
        class MyReader(BaseCorpusReader):
            @classmethod
            def _default_file_pattern(cls) -> str:
                return "*.txt"

            def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
                yield path.read_text(), {"filename": path.name}
    """

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.BASIC,
        metadata_pattern: str = "metadata/*.json",
    ):
        """Initialize the corpus reader.

        Args:
            root: Root directory containing corpus files.
            fileids: Glob pattern for selecting files. If None, uses class default.
            encoding: Text encoding for reading files.
            annotation_level: How much NLP annotation to apply.
            metadata_pattern: Glob pattern for metadata JSON files. Set to None to disable.
        """
        self._root = Path(root).resolve()
        self._fileids_pattern = fileids or self._default_file_pattern()
        self._encoding = encoding
        self._annotation_level = annotation_level
        self._nlp: Language | None = None  # Lazy loaded
        self._metadata_pattern = metadata_pattern
        self._metadata: dict[str, dict[str, Any]] | None = None  # Lazy loaded

    @property
    def root(self) -> Path:
        """Root directory of the corpus."""
        return self._root

    @property
    def nlp(self) -> Language | None:
        """spaCy pipeline (lazy loaded on first access)."""
        if self._nlp is None and self._annotation_level != AnnotationLevel.NONE:
            self._nlp = get_nlp(self._annotation_level)
        return self._nlp

    @property
    def annotation_level(self) -> AnnotationLevel:
        """Current annotation level."""
        return self._annotation_level

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        """Load and aggregate metadata from JSON files.

        Searches for JSON files matching metadata_pattern and merges them
        by fileid key. Later files override earlier ones for duplicate keys.

        Returns:
            Dict mapping fileid -> metadata dict.
        """
        if self._metadata_pattern is None:
            return {}

        merged: dict[str, dict[str, Any]] = {}

        for json_file in sorted(self._root.glob(self._metadata_pattern)):
            try:
                data = json.loads(json_file.read_text(encoding=self._encoding))
                if isinstance(data, dict):
                    for fileid, meta in data.items():
                        if isinstance(meta, dict):
                            merged.setdefault(fileid, {}).update(meta)
            except (json.JSONDecodeError, OSError):
                # Skip malformed or unreadable files
                continue

        return merged

    def get_metadata(self, fileid: str) -> dict[str, Any]:
        """Get metadata for a specific file.

        Args:
            fileid: File identifier.

        Returns:
            Metadata dict for the file, or empty dict if not found.
        """
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata.get(fileid, {})

    def metadata(
        self,
        fileids: str | list[str] | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (fileid, metadata) pairs.

        Args:
            fileids: Files to get metadata for, or None for all.

        Yields:
            Tuples of (fileid, metadata_dict).
        """
        for fileid in self._resolve_fileids(fileids):
            yield fileid, self.get_metadata(fileid)

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Default glob pattern for this corpus type. Override in subclasses."""
        return "*.*"

    def _normalize_text(self, text: str) -> str:
        """Normalize text. Override for corpus-specific cleaning.

        Args:
            text: Raw text from file.

        Returns:
            Normalized text.
        """
        return unicodedata.normalize("NFC", text)

    def fileids(self, match: str | None = None) -> list[str]:
        """Return list of file identifiers matching the pattern.

        Args:
            match: Optional regex pattern to filter filenames.

        Returns:
            Naturally sorted list of matching file identifiers (relative paths).
        """
        from natsort import natsorted

        pattern = self._fileids_pattern
        files = self._root.glob(pattern)

        # Convert to relative paths as strings
        result = [str(f.relative_to(self._root)) for f in files if f.is_file()]

        # Apply optional regex filter
        if match:
            regex = re.compile(match, re.IGNORECASE)
            result = [f for f in result if regex.search(f)]

        # Natural sort (handles numbers correctly: part.1, part.2, ..., part.10)
        return natsorted(result)

    def _resolve_fileids(self, fileids: str | list[str] | None) -> list[str]:
        """Resolve fileids argument to a list of file identifiers.

        Args:
            fileids: Single fileid, list of fileids, or None for all files.

        Returns:
            List of file identifiers.
        """
        if fileids is None:
            return self.fileids()
        if isinstance(fileids, str):
            return [fileids]
        return list(fileids)

    def _iter_paths(self, fileids: str | list[str] | None = None) -> Iterator[Path]:
        """Iterate over file paths for the given fileids.

        Args:
            fileids: Files to iterate over.

        Yields:
            Path objects for each file.
        """
        for fid in self._resolve_fileids(fileids):
            yield self._root / fid

    @abstractmethod
    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a single file. Yield (text_chunk, metadata) pairs.

        This is the main extension point for subclasses. Implement this method
        to handle the specific file format of your corpus.

        Args:
            path: Path to the file to parse.

        Yields:
            Tuples of (text, metadata_dict) for each logical unit in the file.
        """
        ...

    # -------------------------------------------------------------------------
    # Core iteration methods
    # -------------------------------------------------------------------------

    def texts(self, fileids: str | list[str] | None = None) -> Iterator[str]:
        """Yield raw text strings. Zero NLP overhead.

        This is the fastest way to iterate over corpus content when you
        don't need any linguistic annotation.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Raw text strings.
        """
        for path in self._iter_paths(fileids):
            for text, _metadata in self._parse_file(path):
                yield self._normalize_text(text)

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Doc objects with annotations.

        The level of annotation depends on the reader's annotation_level setting.
        Metadata from JSON files is merged with any metadata from _parse_file().

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot create Docs with annotation_level=NONE. "
                "Use texts() for raw strings, or set a higher annotation level."
            )

        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))
            # Get JSON metadata and merge with file-level metadata
            json_metadata = self.get_metadata(fileid)

            for text, file_metadata in self._parse_file(path):
                text = self._normalize_text(text)
                doc = nlp(text)
                doc._.fileid = fileid
                # Merge: JSON metadata as base, file metadata overrides
                doc._.metadata = {**json_metadata, **file_metadata}
                yield doc

    def sents(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Span | str"]:
        """Yield sentences from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Span objects.

        Yields:
            Sentence Spans (or strings if as_text=True).
        """
        for doc in self.docs(fileids):
            for sent in doc.sents:
                yield sent.text if as_text else sent

    def tokens(
        self,
        fileids: str | list[str] | None = None,
        as_text: bool = False,
    ) -> Iterator["Token | str"]:
        """Yield individual tokens from documents.

        Args:
            fileids: Files to process, or None for all.
            as_text: If True, yield strings instead of Token objects.

        Yields:
            Tokens (or strings if as_text=True).
        """
        for doc in self.docs(fileids):
            for token in doc:
                yield token.text if as_text else token

    # -------------------------------------------------------------------------
    # Text analysis methods
    # -------------------------------------------------------------------------

    def _get_token_citation(self, doc: "Doc", token: "Token", token_idx: int) -> str:
        """Get citation for a token, checking spans if not on token directly.

        Args:
            doc: The document containing the token.
            token: The token to get citation for.
            token_idx: Index of the token in the document.

        Returns:
            Citation string, or fileid:idx fallback.
        """
        # First check token-level citation
        citation = getattr(token._, "citation", None)
        if citation is not None:
            return citation

        # Check if token is within a span that has a citation (e.g., Tesserae lines)
        for span_key in doc.spans:
            for span in doc.spans[span_key]:
                if span.start <= token.i < span.end:
                    span_citation = getattr(span._, "citation", None)
                    if span_citation is not None:
                        return span_citation

        # Fallback to fileid:index
        fileid = doc._.fileid or "unknown"
        return f"{fileid}:{token_idx}"

    def concordance(
        self,
        fileids: str | list[str] | None = None,
        basis: str = "lemma",
        only_alpha: bool = True,
    ) -> dict[str, list[str]]:
        """Build a concordance mapping words to their citation locations.

        A concordance is a dictionary where keys are word forms and values
        are lists of citations/locations where that word appears.

        Args:
            fileids: Files to process, or None for all.
            basis: How to key the concordance:
                - "lemma": group by lemma (default, recommended)
                - "norm": group by normalized form (spaCy's norm_)
                - "text": group by exact surface form
            only_alpha: If True, skip non-alphabetic tokens (punctuation, numbers).

        Returns:
            Dict mapping word form -> list of citation strings.
            Citations are in format "<citation>" if available, else "fileid:token_idx".

        Example:
            >>> conc = reader.concordance(basis="lemma")
            >>> conc["amor"]
            ['<catull. 1.1>', '<catull. 1.3>', '<verg. aen. 4.1>']
        """
        from collections import defaultdict

        concordance_dict: defaultdict[str, list[str]] = defaultdict(list)

        for doc in self.docs(fileids):
            for i, token in enumerate(doc):
                # Skip non-alphabetic tokens if requested
                if only_alpha and not token.is_alpha:
                    continue

                # Determine the key based on basis
                if basis == "lemma":
                    key = token.lemma_
                elif basis == "norm":
                    key = token.norm_
                else:  # "text" or fallback
                    key = token.text

                citation = self._get_token_citation(doc, token, i)
                concordance_dict[key].append(citation)

        # Sort by key and return as regular dict
        return dict(sorted(concordance_dict.items()))

    def kwic(
        self,
        keyword: str,
        fileids: str | list[str] | None = None,
        window: int = 5,
        ignore_case: bool = True,
        by_lemma: bool = False,
        limit: int | None = None,
    ) -> Iterator[dict[str, str]]:
        """Find keyword in context (KWIC) across the corpus.

        Returns matches with surrounding context, useful for studying
        word usage patterns.

        Args:
            keyword: Word to search for.
            fileids: Files to search, or None for all.
            window: Number of tokens on each side for context.
            ignore_case: If True, match case-insensitively.
            by_lemma: If True, match against lemma instead of surface form.
            limit: Maximum number of results to return.

        Yields:
            Dicts with keys:
                - left: left context (string)
                - match: matched token (string)
                - right: right context (string)
                - citation: citation string if available
                - fileid: file identifier

        Example:
            >>> for hit in reader.kwic("amor", window=3, by_lemma=True):
            ...     print(f"{hit['left']} [{hit['match']}] {hit['right']}")
            ...     print(f"  -- {hit['citation']}")
        """
        target = keyword.lower() if ignore_case else keyword
        count = 0

        for doc in self.docs(fileids):
            fileid = doc._.fileid or "unknown"
            tokens = list(doc)

            for i, token in enumerate(tokens):
                # Determine what to match against
                if by_lemma:
                    token_value = token.lemma_.lower() if ignore_case else token.lemma_
                else:
                    token_value = token.text.lower() if ignore_case else token.text

                if token_value == target:
                    # Build context windows
                    left_start = max(0, i - window)
                    right_end = min(len(tokens), i + window + 1)

                    left_tokens = tokens[left_start:i]
                    right_tokens = tokens[i + 1:right_end]

                    left_text = " ".join(t.text for t in left_tokens)
                    right_text = " ".join(t.text for t in right_tokens)

                    citation = self._get_token_citation(doc, token, i)

                    yield {
                        "left": left_text,
                        "match": token.text,
                        "right": right_text,
                        "citation": citation,
                        "fileid": fileid,
                    }

                    count += 1
                    if limit is not None and count >= limit:
                        return
