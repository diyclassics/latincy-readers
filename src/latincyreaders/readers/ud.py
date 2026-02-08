"""Universal Dependencies (CONLLU/CONLLUP) corpus reader.

Reads texts in CoNLL-U format (and the CONLLUP variant used by LASLA).
By default, annotations (lemma, POS, morphology) are taken directly from
the file rather than running through LatinCy.

CoNLL-U format (10 tab-separated columns):
    1. ID - Word index
    2. FORM - Word form
    3. LEMMA - Lemma
    4. UPOS - Universal POS tag
    5. XPOS - Language-specific POS tag
    6. FEATS - Morphological features
    7. HEAD - Head word index (empty in CONLLUP)
    8. DEPREL - Dependency relation (empty in CONLLUP)
    9. DEPS - Enhanced dependencies (empty in CONLLUP)
    10. MISC - Miscellaneous

CONLLUP (LASLA format) is identical but columns 7-9 are typically empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

import spacy
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab

from latincyreaders.core.base import BaseCorpusReader, AnnotationLevel
from latincyreaders.nlp.pipeline import get_nlp

if TYPE_CHECKING:
    from spacy.tokens import Span


# Register UD-specific token extensions
def _register_ud_extensions() -> None:
    """Register spaCy custom extensions for UD annotations."""
    extensions = [
        ("ud_id", None),      # Original token ID from file
        ("ud_lemma", None),   # Lemma from UD file
        ("ud_upos", None),    # UPOS from UD file
        ("ud_xpos", None),    # XPOS from UD file
        ("ud_feats", None),   # Morphological features dict
        ("ud_head", None),    # Head index from UD file
        ("ud_deprel", None),  # Dependency relation from UD file
        ("ud_deps", None),    # Enhanced dependencies
        ("ud_misc", None),    # Miscellaneous field
    ]
    for name, default in extensions:
        if not Token.has_extension(name):
            Token.set_extension(name, default=default)


_register_ud_extensions()


@dataclass
class UDToken:
    """A single token from a CoNLL-U/CONLLUP file."""

    id: str  # Can be "1", "1-2" for MWT, or "1.1" for empty nodes
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: dict[str, str] = field(default_factory=dict)
    head: str | None = None
    deprel: str | None = None
    deps: str | None = None
    misc: dict[str, str] = field(default_factory=dict)

    @property
    def is_multiword(self) -> bool:
        """Check if this is a multi-word token range (e.g., '1-2')."""
        return "-" in self.id

    @property
    def is_empty_node(self) -> bool:
        """Check if this is an empty node (e.g., '1.1')."""
        return "." in self.id


@dataclass
class UDSentence:
    """A sentence from a CoNLL-U/CONLLUP file."""

    tokens: list[UDToken]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Reconstruct sentence text from tokens."""
        # Use sent_text from metadata if available
        if "text" in self.metadata:
            return self.metadata["text"]
        # Otherwise join non-MWT tokens
        return " ".join(
            t.form for t in self.tokens
            if not t.is_multiword and not t.is_empty_node
        )

    @property
    def sent_id(self) -> str | None:
        """Get sentence ID from metadata."""
        return self.metadata.get("sent_id")


class UDReader(BaseCorpusReader):
    """Reader for CoNLL-U and CONLLUP format Latin texts.

    By default, annotations (lemma, POS, morphology) are taken directly
    from the file. Set use_file_annotations=False to use LatinCy instead,
    in which case the original UD annotations are stored in Token._.ud_*
    extensions.

    Supports both:
    - CoNLL-U (.conllu): Full UD format with dependency annotations
    - CONLLUP (.conllup): LASLA variant without dependency columns

    Example:
        >>> reader = UDReader("/path/to/corpus")
        >>> for doc in reader.docs():
        ...     for token in doc:
        ...         print(f"{token.text}: {token.lemma_} ({token.pos_})")

        >>> # Use LatinCy annotations, keep UD originals
        >>> reader = UDReader("/path/to/corpus", use_file_annotations=False)
        >>> for doc in reader.docs():
        ...     for token in doc:
        ...         print(f"LatinCy: {token.lemma_}, UD: {token._.ud_lemma}")

    Attributes:
        use_file_annotations: If True (default), use annotations from file.
            If False, use LatinCy and store file annotations in Token._.ud_*.
    """

    COMMENT_PATTERN = re.compile(r"^#\s*(\S+)\s*=\s*(.*)$")

    def __init__(
        self,
        root: str | Path,
        fileids: str | None = None,
        encoding: str = "utf-8",
        annotation_level: AnnotationLevel = AnnotationLevel.FULL,
        use_file_annotations: bool = True,
        cache: bool = True,
        cache_maxsize: int = 128,
    ):
        """Initialize the UD reader.

        Args:
            root: Root directory containing .conllu/.conllup files.
            fileids: Glob pattern for selecting files.
            encoding: Text encoding.
            annotation_level: How much NLP annotation to apply (only used
                when use_file_annotations=False).
            use_file_annotations: If True (default), use lemma/POS/morph
                from the file. If False, use LatinCy and store originals
                in Token._.ud_* extensions.
            cache: If True (default), cache processed Doc objects.
            cache_maxsize: Maximum number of documents to cache.
        """
        super().__init__(
            root, fileids, encoding, annotation_level,
            cache=cache, cache_maxsize=cache_maxsize
        )
        self._use_file_annotations = use_file_annotations
        self._vocab: Vocab | None = None

    @property
    def use_file_annotations(self) -> bool:
        """Whether to use annotations from the UD file."""
        return self._use_file_annotations

    @property
    def vocab(self) -> Vocab:
        """Get spaCy vocab for creating Docs."""
        if self._vocab is None:
            # Use LatinCy vocab if available, otherwise blank Latin
            if self._use_file_annotations:
                try:
                    nlp = spacy.load("la_core_web_lg")
                    self._vocab = nlp.vocab
                except OSError:
                    self._vocab = spacy.blank("la").vocab
            else:
                nlp = get_nlp(self._annotation_level)
                self._vocab = nlp.vocab if nlp else spacy.blank("la").vocab
        return self._vocab

    @classmethod
    def _default_file_pattern(cls) -> str:
        """Match both .conllu and .conllup files."""
        return "**/*.conll*"

    def _parse_feats(self, feats_str: str) -> dict[str, str]:
        """Parse morphological features string into dict.

        Args:
            feats_str: Features in Key=Value|Key=Value format.

        Returns:
            Dict of feature key-value pairs.
        """
        if feats_str == "_" or not feats_str:
            return {}
        result = {}
        for pair in feats_str.split("|"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key] = value
        return result

    def _parse_misc(self, misc_str: str) -> dict[str, str]:
        """Parse MISC field into dict."""
        return self._parse_feats(misc_str)

    def _parse_token_line(self, line: str) -> UDToken | None:
        """Parse a single token line from CoNLL-U format.

        Args:
            line: Tab-separated token line.

        Returns:
            UDToken object, or None if line is invalid.
        """
        parts = line.split("\t")
        if len(parts) < 10:
            # CONLLUP may have fewer columns; pad with underscores
            parts.extend(["_"] * (10 - len(parts)))

        try:
            return UDToken(
                id=parts[0],
                form=parts[1],
                lemma=parts[2] if parts[2] != "_" else parts[1],
                upos=parts[3] if parts[3] != "_" else "",
                xpos=parts[4] if parts[4] != "_" else "",
                feats=self._parse_feats(parts[5]),
                head=parts[6] if parts[6] != "_" else None,
                deprel=parts[7] if parts[7] != "_" else None,
                deps=parts[8] if parts[8] != "_" else None,
                misc=self._parse_misc(parts[9]),
            )
        except (IndexError, ValueError):
            return None

    def _parse_sentences(self, text: str) -> Iterator[UDSentence]:
        """Parse CoNLL-U/CONLLUP text into sentences.

        Args:
            text: Raw file content.

        Yields:
            UDSentence objects.
        """
        current_tokens: list[UDToken] = []
        current_metadata: dict[str, str] = {}

        for line in text.split("\n"):
            line = line.rstrip()

            if not line:
                # Blank line = end of sentence
                if current_tokens:
                    yield UDSentence(
                        tokens=current_tokens,
                        metadata=current_metadata,
                    )
                    current_tokens = []
                    current_metadata = {}
                continue

            if line.startswith("#"):
                # Comment/metadata line
                match = self.COMMENT_PATTERN.match(line)
                if match:
                    current_metadata[match.group(1)] = match.group(2).strip()
                continue

            # Token line
            token = self._parse_token_line(line)
            if token:
                current_tokens.append(token)

        # Handle file not ending with blank line
        if current_tokens:
            yield UDSentence(
                tokens=current_tokens,
                metadata=current_metadata,
            )

    def _parse_file(self, path: Path) -> Iterator[tuple[str, dict]]:
        """Parse a CoNLL-U/CONLLUP file.

        Args:
            path: Path to file.

        Yields:
            (text, metadata) tuples for each file.
        """
        raw_text = path.read_text(encoding=self._encoding)
        sentences = list(self._parse_sentences(raw_text))

        if not sentences:
            return

        # Combine sentence texts
        combined_text = " ".join(sent.text for sent in sentences)

        metadata = {
            "filename": path.name,
            "path": str(path),
            "format": "conllup" if path.suffix == ".conllup" else "conllu",
            "n_sentences": len(sentences),
            "_sentences": sentences,  # Private: used for Doc creation
        }

        yield combined_text, metadata

    def _create_doc_from_sentences(
        self,
        sentences: list[UDSentence],
        fileid: str,
    ) -> "Doc":
        """Create a spaCy Doc directly from UD sentences.

        This preserves the exact tokenization and annotations from the
        CoNLL-U/CONLLUP file.

        Args:
            sentences: List of parsed UDSentence objects.
            fileid: File identifier.

        Returns:
            spaCy Doc with UD annotations.
        """
        words: list[str] = []
        spaces: list[bool] = []
        sent_starts: list[bool] = []

        # Collect token data for Doc creation
        token_data: list[UDToken] = []

        for sent_idx, sent in enumerate(sentences):
            # Filter out MWT range tokens and empty nodes for Doc
            real_tokens = [
                t for t in sent.tokens
                if not t.is_multiword and not t.is_empty_node
            ]

            for tok_idx, token in enumerate(real_tokens):
                words.append(token.form)
                # Check MISC for SpaceAfter=No
                space_after = token.misc.get("SpaceAfter", "Yes") != "No"
                # Last token in sentence typically has space (unless SpaceAfter=No)
                if tok_idx == len(real_tokens) - 1 and sent_idx < len(sentences) - 1:
                    spaces.append(space_after)
                else:
                    spaces.append(space_after)

                # Mark sentence boundaries
                sent_starts.append(tok_idx == 0)
                token_data.append(token)

        # Create Doc with words and spaces
        doc = Doc(self.vocab, words=words, spaces=spaces, sent_starts=sent_starts)

        # Apply annotations from UD file
        for i, (token, ud_token) in enumerate(zip(doc, token_data)):
            # Set standard spaCy attributes
            token.lemma_ = ud_token.lemma
            if ud_token.upos:
                token.pos_ = ud_token.upos
            if ud_token.xpos:
                token.tag_ = ud_token.xpos

            # Set UD-specific extensions (always, for reference)
            token._.ud_id = ud_token.id
            token._.ud_lemma = ud_token.lemma
            token._.ud_upos = ud_token.upos
            token._.ud_xpos = ud_token.xpos
            token._.ud_feats = ud_token.feats
            token._.ud_head = ud_token.head
            token._.ud_deprel = ud_token.deprel
            token._.ud_deps = ud_token.deps
            token._.ud_misc = ud_token.misc

            # Set morphology from feats
            if ud_token.feats:
                morph_str = "|".join(f"{k}={v}" for k, v in ud_token.feats.items())
                token.set_morph(morph_str)

        doc._.fileid = fileid
        return doc

    def _create_doc_with_latincy(
        self,
        sentences: list[UDSentence],
        fileid: str,
    ) -> "Doc":
        """Create a Doc using LatinCy, storing UD annotations in extensions.

        Args:
            sentences: List of parsed UDSentence objects.
            fileid: File identifier.

        Returns:
            spaCy Doc with LatinCy annotations and UD originals in _.ud_*.
        """
        nlp = self.nlp
        if nlp is None:
            raise ValueError(
                "Cannot use LatinCy without NLP pipeline. "
                "Set annotation_level > NONE or use_file_annotations=True."
            )

        # Get text and process with LatinCy
        text = " ".join(sent.text for sent in sentences)
        doc = nlp(text)
        doc._.fileid = fileid

        # Build a mapping of character positions to UD tokens
        # This is approximate since tokenization may differ
        ud_tokens: list[UDToken] = []
        for sent in sentences:
            for token in sent.tokens:
                if not token.is_multiword and not token.is_empty_node:
                    ud_tokens.append(token)

        # Try to align UD tokens with spaCy tokens by form matching
        ud_idx = 0
        for spacy_token in doc:
            if ud_idx >= len(ud_tokens):
                break

            ud_token = ud_tokens[ud_idx]

            # Check for match (exact or case-insensitive)
            if (spacy_token.text == ud_token.form or
                spacy_token.text.lower() == ud_token.form.lower()):
                # Store UD annotations in extensions
                spacy_token._.ud_id = ud_token.id
                spacy_token._.ud_lemma = ud_token.lemma
                spacy_token._.ud_upos = ud_token.upos
                spacy_token._.ud_xpos = ud_token.xpos
                spacy_token._.ud_feats = ud_token.feats
                spacy_token._.ud_head = ud_token.head
                spacy_token._.ud_deprel = ud_token.deprel
                spacy_token._.ud_deps = ud_token.deps
                spacy_token._.ud_misc = ud_token.misc
                ud_idx += 1
            # Handle cases where spaCy splits differently
            elif ud_token.form.startswith(spacy_token.text):
                # spaCy token is prefix of UD token - store partial match
                spacy_token._.ud_lemma = ud_token.lemma
                spacy_token._.ud_upos = ud_token.upos
                # Don't advance ud_idx until we consume the full token

        return doc

    def docs(self, fileids: str | list[str] | None = None) -> Iterator["Doc"]:
        """Yield spaCy Docs with UD annotations.

        Annotations come from the file by default, or from LatinCy if
        use_file_annotations=False.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            spaCy Doc objects with annotations.
        """
        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))

            # Check cache
            if self._cache_enabled and fileid in self._cache:
                self._cache_hits += 1
                self._cache.move_to_end(fileid)
                yield self._cache[fileid]
                continue

            if self._cache_enabled:
                self._cache_misses += 1

            # Get JSON metadata
            json_metadata = self.get_metadata(fileid)

            for text, file_metadata in self._parse_file(path):
                sentences = file_metadata.get("_sentences", [])

                if self._use_file_annotations:
                    doc = self._create_doc_from_sentences(sentences, fileid)
                else:
                    doc = self._create_doc_with_latincy(sentences, fileid)

                # Merge metadata
                clean_file_meta = {
                    k: v for k, v in file_metadata.items()
                    if not k.startswith("_")
                }
                doc._.metadata = {**json_metadata, **clean_file_meta}

                # Create sentence spans with sent_id citations
                sent_spans = []
                for sent_idx, ud_sent in enumerate(sentences):
                    if sent_idx < len(list(doc.sents)):
                        spacy_sent = list(doc.sents)[sent_idx]
                        if ud_sent.sent_id:
                            spacy_sent._.citation = ud_sent.sent_id
                        sent_spans.append(spacy_sent)
                doc.spans["sentences"] = sent_spans

                # Cache if enabled
                if self._cache_enabled:
                    while len(self._cache) >= self._cache_maxsize:
                        self._cache.popitem(last=False)
                    self._cache[fileid] = doc

                yield doc

    def sentences(
        self,
        fileids: str | list[str] | None = None,
    ) -> Iterator["Span"]:
        """Yield sentence spans with UD metadata.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Span objects for each sentence.
        """
        for doc in self.docs(fileids):
            yield from doc.spans.get("sentences", doc.sents)

    def tokens_with_annotations(
        self,
        fileids: str | list[str] | None = None,
    ) -> Iterator[dict]:
        """Yield token dicts with full UD annotations.

        Useful for accessing all UD fields without spaCy Doc overhead.

        Args:
            fileids: Files to process, or None for all.

        Yields:
            Dicts with token data including all UD fields.
        """
        for path in self._iter_paths(fileids):
            fileid = str(path.relative_to(self._root))
            raw_text = path.read_text(encoding=self._encoding)

            for sent in self._parse_sentences(raw_text):
                for token in sent.tokens:
                    if token.is_multiword or token.is_empty_node:
                        continue
                    yield {
                        "fileid": fileid,
                        "sent_id": sent.sent_id,
                        "id": token.id,
                        "form": token.form,
                        "lemma": token.lemma,
                        "upos": token.upos,
                        "xpos": token.xpos,
                        "feats": token.feats,
                        "head": token.head,
                        "deprel": token.deprel,
                        "deps": token.deps,
                        "misc": token.misc,
                    }

    def has_dependencies(self, fileids: str | list[str] | None = None) -> bool:
        """Check if files have dependency annotations (CONLLU vs CONLLUP).

        Args:
            fileids: Files to check, or None for all.

        Returns:
            True if any file has non-empty HEAD/DEPREL columns.
        """
        for path in self._iter_paths(fileids):
            raw_text = path.read_text(encoding=self._encoding)
            for sent in self._parse_sentences(raw_text):
                for token in sent.tokens:
                    if token.head is not None or token.deprel is not None:
                        return True
        return False
