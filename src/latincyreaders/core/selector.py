"""FileSelector fluent API for filtering corpus files.

This module provides a fluent API for building complex file queries
with metadata filtering, as a complement to fileids(match=...).
"""

from __future__ import annotations

import re
from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from latincyreaders.core.base import BaseCorpusReader


class FileSelector:
    """Fluent API for selecting corpus files by filename and metadata.

    FileSelector provides a chainable interface for building complex
    file queries. It supports:
    - Regex matching on filenames
    - Exact metadata matching
    - __in operator for matching any of multiple values
    - Numeric range filtering

    Example:
        >>> # Select all epic poetry by Vergil or Ovid from 50 BCE to 50 CE
        >>> selection = reader.select()
        ...     .where(genre="epic")
        ...     .where(author__in=["Vergil", "Ovid"])
        ...     .date_range(-50, 50)
        >>> for doc in reader.docs(selection):
        ...     print(doc._.fileid)

    Note:
        FileSelector is immutable - each method returns a new FileSelector
        instance, allowing safe chaining without side effects.
    """

    # Supported operators for where() kwargs
    _VALID_OPERATORS = frozenset({"in"})

    def __init__(
        self,
        reader: "BaseCorpusReader",
        *,
        _match_patterns: tuple[str, ...] = (),
        _where_exact: dict[str, Any] | None = None,
        _where_in: dict[str, list[Any]] | None = None,
        _where_between: list[tuple[str, int | float, int | float]] | None = None,
    ):
        """Initialize a FileSelector.

        Args:
            reader: The corpus reader to select files from.
            _match_patterns: Internal - accumulated regex patterns.
            _where_exact: Internal - exact match filters.
            _where_in: Internal - __in operator filters.
            _where_between: Internal - numeric range filters.
        """
        self._reader = reader
        self._match_patterns = _match_patterns
        self._where_exact = _where_exact or {}
        self._where_in = _where_in or {}
        self._where_between = _where_between or []

    def match(self, pattern: str) -> "FileSelector":
        """Filter files by regex pattern on filename.

        Args:
            pattern: Regex pattern to match against filenames (case-insensitive).

        Returns:
            New FileSelector with the pattern added.

        Example:
            >>> reader.select().match("vergil").match("aeneid")
        """
        return FileSelector(
            self._reader,
            _match_patterns=self._match_patterns + (pattern,),
            _where_exact=self._where_exact.copy(),
            _where_in=self._where_in.copy(),
            _where_between=self._where_between.copy(),
        )

    def where(self, **kwargs: Any) -> "FileSelector":
        """Filter files by metadata field values.

        Supports two forms:
        - Exact match: where(field=value) matches files where metadata[field] == value
        - Any of: where(field__in=[v1, v2]) matches files where metadata[field] in [v1, v2]

        Multiple conditions are ANDed together.

        Args:
            **kwargs: Field=value or field__op=value filters.

        Returns:
            New FileSelector with the filters added.

        Raises:
            ValueError: If an unknown operator is used or conflicting filters exist.

        Example:
            >>> reader.select().where(author="Vergil", genre="epic")
            >>> reader.select().where(author__in=["Vergil", "Ovid"])
        """
        new_exact = self._where_exact.copy()
        new_in = self._where_in.copy()

        for key, value in kwargs.items():
            if "__" in key:
                field, op = key.rsplit("__", 1)
                if op not in self._VALID_OPERATORS:
                    raise ValueError(
                        f"Unknown operator '{op}' in '{key}'. "
                        f"Valid operators: {', '.join(sorted(self._VALID_OPERATORS))}"
                    )
                if op == "in":
                    # Check for conflict with existing exact match
                    if field in new_exact:
                        raise ValueError(
                            f"Conflicting filters for field '{field}': "
                            f"cannot use both exact match and __in operator"
                        )
                    new_in[field] = list(value)
            else:
                # Exact match
                # Check for conflict with existing __in
                if key in new_in:
                    raise ValueError(
                        f"Conflicting filters for field '{key}': "
                        f"cannot use both exact match and __in operator"
                    )
                new_exact[key] = value

        return FileSelector(
            self._reader,
            _match_patterns=self._match_patterns,
            _where_exact=new_exact,
            _where_in=new_in,
            _where_between=self._where_between.copy(),
        )

    def where_between(
        self, field: str, start: int | float, end: int | float
    ) -> "FileSelector":
        """Filter files by numeric range on a metadata field.

        The range is inclusive on both ends: start <= value <= end.

        Args:
            field: Metadata field name.
            start: Minimum value (inclusive).
            end: Maximum value (inclusive).

        Returns:
            New FileSelector with the range filter added.

        Example:
            >>> reader.select().where_between("lines", 100, 500)
        """
        return FileSelector(
            self._reader,
            _match_patterns=self._match_patterns,
            _where_exact=self._where_exact.copy(),
            _where_in=self._where_in.copy(),
            _where_between=self._where_between + [(field, start, end)],
        )

    def date_range(self, start: int, end: int) -> "FileSelector":
        """Filter files by date range.

        Convenience method equivalent to where_between("date", start, end).
        Uses negative numbers for BCE dates: -50 = 50 BCE.

        Args:
            start: Start year (inclusive).
            end: End year (inclusive).

        Returns:
            New FileSelector with the date filter added.

        Example:
            >>> reader.select().date_range(-50, 50)  # 50 BCE to 50 CE
        """
        return self.where_between("date", start, end)

    def _apply_filters(self) -> Iterator[str]:
        """Apply all filters and yield matching file IDs.

        Yields:
            File IDs that pass all filters.
        """
        # Start with all file IDs
        for fileid in self._reader.fileids():
            # Apply match patterns
            if not self._passes_match_filters(fileid):
                continue

            # Apply metadata filters
            if not self._passes_metadata_filters(fileid):
                continue

            yield fileid

    def _passes_match_filters(self, fileid: str) -> bool:
        """Check if a file ID passes all match patterns."""
        for pattern in self._match_patterns:
            if not re.search(pattern, fileid, re.IGNORECASE):
                return False
        return True

    def _passes_metadata_filters(self, fileid: str) -> bool:
        """Check if a file ID passes all metadata filters."""
        # Skip metadata lookup if no filters
        if not self._where_exact and not self._where_in and not self._where_between:
            return True

        metadata = self._reader.get_metadata(fileid)

        # Check exact matches
        for field, value in self._where_exact.items():
            if field not in metadata:
                return False
            if metadata[field] != value:
                return False

        # Check __in matches
        for field, values in self._where_in.items():
            if field not in metadata:
                return False
            if metadata[field] not in values:
                return False

        # Check range matches
        for field, start, end in self._where_between:
            if field not in metadata:
                return False
            field_value = metadata[field]
            if not isinstance(field_value, (int, float)):
                return False
            if not (start <= field_value <= end):
                return False

        return True

    def preview(self, n: int = 10) -> list[str]:
        """Return first n matching file IDs.

        Args:
            n: Maximum number of results to return.

        Returns:
            List of up to n matching file IDs.
        """
        result = []
        for fileid in self._apply_filters():
            result.append(fileid)
            if len(result) >= n:
                break
        return result

    def count(self) -> int:
        """Return the count of matching files.

        Returns:
            Number of files matching all filters.
        """
        return len(self)

    def to_list(self) -> list[str]:
        """Materialize all matching file IDs to a list.

        Returns:
            List of all matching file IDs.
        """
        return list(self)

    def __iter__(self) -> Iterator[str]:
        """Iterate over matching file IDs.

        Yields:
            File IDs that pass all filters.
        """
        return self._apply_filters()

    def __len__(self) -> int:
        """Return the count of matching files.

        Returns:
            Number of files matching all filters.
        """
        return sum(1 for _ in self._apply_filters())
