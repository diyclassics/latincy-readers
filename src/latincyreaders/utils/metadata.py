"""Metadata management with schema validation.

This module provides the MetadataManager class for loading, validating,
and querying corpus metadata from JSON files.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class MetadataSchema:
    """Schema definition for metadata validation.

    Defines required and optional fields with their expected types.

    Example:
        >>> schema = MetadataSchema(
        ...     required={"author": str, "title": str},
        ...     optional={"date": int, "genre": str},
        ... )
    """

    required: dict[str, type] = field(default_factory=dict)
    optional: dict[str, type] = field(default_factory=dict)

    def validate(self, fileid: str, metadata: dict[str, Any]) -> list[str]:
        """Validate metadata against this schema.

        Args:
            fileid: File identifier (for error messages).
            metadata: Metadata dict to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Check required fields
        for field_name, expected_type in self.required.items():
            if field_name not in metadata:
                errors.append(f"{fileid}: missing required field '{field_name}'")
            elif not isinstance(metadata[field_name], expected_type):
                actual = type(metadata[field_name]).__name__
                expected = expected_type.__name__
                errors.append(
                    f"{fileid}: field '{field_name}' has type {actual}, expected {expected}"
                )

        # Check optional field types (if present)
        for field_name, expected_type in self.optional.items():
            if field_name in metadata and not isinstance(metadata[field_name], expected_type):
                actual = type(metadata[field_name]).__name__
                expected = expected_type.__name__
                errors.append(
                    f"{fileid}: field '{field_name}' has type {actual}, expected {expected}"
                )

        return errors


# Common schema for Latin corpus metadata
LATIN_CORPUS_SCHEMA = MetadataSchema(
    required={},  # No required fields by default
    optional={
        "author": str,
        "title": str,
        "date": int,  # Year (negative for BCE)
        "genre": str,
        "work": str,
        "book": (int, str),  # Can be int or str
        "collection": str,
    },
)


@dataclass
class ValidationResult:
    """Result of metadata validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


class MetadataManager:
    """Manages corpus metadata with schema validation.

    Loads metadata from JSON files, validates against schema, and provides
    query methods for filtering files by metadata fields.

    Example:
        >>> manager = MetadataManager(Path("/corpus"), schema=LATIN_CORPUS_SCHEMA)
        >>> result = manager.validate()
        >>> if not result:
        ...     for error in result.errors:
        ...         print(error)

        >>> # Get metadata for a file
        >>> meta = manager.get("vergil.aen.tess")
        >>> print(meta.get("author"))

        >>> # Filter files by metadata
        >>> for fid in manager.filter_by(author="Vergil"):
        ...     print(fid)
    """

    def __init__(
        self,
        root: Path,
        pattern: str = "metadata/*.json",
        encoding: str = "utf-8",
        schema: MetadataSchema | None = None,
    ):
        """Initialize the MetadataManager.

        Args:
            root: Root directory containing metadata files.
            pattern: Glob pattern for finding metadata JSON files.
            encoding: Text encoding for reading files.
            schema: Optional schema for validation. If None, no validation.
        """
        self._root = Path(root)
        self._pattern = pattern
        self._encoding = encoding
        self._schema = schema
        self._metadata: dict[str, dict[str, Any]] | None = None
        self._load_errors: list[str] = []

    @property
    def metadata(self) -> dict[str, dict[str, Any]]:
        """Lazy-loaded metadata dictionary."""
        if self._metadata is None:
            self._metadata = self._load()
        return self._metadata

    def _load(self) -> dict[str, dict[str, Any]]:
        """Load and merge all metadata JSON files.

        Returns:
            Dict mapping fileid -> metadata dict.
        """
        merged: dict[str, dict[str, Any]] = {}
        self._load_errors = []

        for json_file in sorted(self._root.glob(self._pattern)):
            try:
                data = json.loads(json_file.read_text(encoding=self._encoding))
                if isinstance(data, dict):
                    for fileid, meta in data.items():
                        if isinstance(meta, dict):
                            merged.setdefault(fileid, {}).update(meta)
                        else:
                            self._load_errors.append(
                                f"{json_file.name}: value for '{fileid}' is not a dict"
                            )
                else:
                    self._load_errors.append(
                        f"{json_file.name}: root is not a dict"
                    )
            except json.JSONDecodeError as e:
                self._load_errors.append(f"{json_file.name}: JSON parse error: {e}")
            except OSError as e:
                self._load_errors.append(f"{json_file.name}: read error: {e}")

        return merged

    def reload(self) -> None:
        """Force reload of metadata from disk."""
        self._metadata = None
        _ = self.metadata  # Trigger reload

    def get(self, fileid: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get metadata for a specific file.

        Args:
            fileid: File identifier.
            default: Default value if not found.

        Returns:
            Metadata dict for the file.
        """
        return self.metadata.get(fileid, default or {})

    def __getitem__(self, fileid: str) -> dict[str, Any]:
        """Get metadata by subscript access.

        Raises:
            KeyError: If fileid not found.
        """
        if fileid not in self.metadata:
            raise KeyError(f"No metadata for '{fileid}'")
        return self.metadata[fileid]

    def __contains__(self, fileid: str) -> bool:
        """Check if metadata exists for a file."""
        return fileid in self.metadata

    def __iter__(self) -> Iterator[str]:
        """Iterate over fileids with metadata."""
        return iter(self.metadata)

    def __len__(self) -> int:
        """Number of files with metadata."""
        return len(self.metadata)

    def items(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Iterate over (fileid, metadata) pairs."""
        return iter(self.metadata.items())

    def validate(self, strict: bool = False) -> ValidationResult:
        """Validate all metadata against the schema.

        Args:
            strict: If True, treat warnings as errors.

        Returns:
            ValidationResult with errors and warnings.
        """
        errors = list(self._load_errors)
        validation_warnings = []

        if self._schema is None:
            return ValidationResult(is_valid=len(errors) == 0, errors=errors)

        for fileid, meta in self.metadata.items():
            field_errors = self._schema.validate(fileid, meta)
            errors.extend(field_errors)

        is_valid = len(errors) == 0
        if strict:
            errors.extend(validation_warnings)
            is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=validation_warnings,
        )

    def validate_or_warn(self) -> bool:
        """Validate and emit warnings for any issues.

        Returns:
            True if valid, False otherwise.
        """
        result = self.validate()
        for error in result.errors:
            warnings.warn(f"Metadata validation error: {error}", stacklevel=2)
        for warning in result.warnings:
            warnings.warn(f"Metadata warning: {warning}", stacklevel=2)
        return result.is_valid

    def filter_by(self, **kwargs: Any) -> Iterator[str]:
        """Filter fileids by metadata field values.

        Args:
            **kwargs: Field=value pairs to match.

        Yields:
            Fileids matching all criteria.

        Example:
            >>> for fid in manager.filter_by(author="Vergil", genre="epic"):
            ...     print(fid)
        """
        for fileid, meta in self.metadata.items():
            if all(meta.get(k) == v for k, v in kwargs.items()):
                yield fileid

    def filter_by_range(
        self,
        field: str,
        min_val: int | float | None = None,
        max_val: int | float | None = None,
    ) -> Iterator[str]:
        """Filter fileids by numeric range on a field.

        Args:
            field: Metadata field name.
            min_val: Minimum value (inclusive), or None for no minimum.
            max_val: Maximum value (inclusive), or None for no maximum.

        Yields:
            Fileids with field value in range.

        Example:
            >>> # Files from 50 BCE to 50 CE
            >>> for fid in manager.filter_by_range("date", -50, 50):
            ...     print(fid)
        """
        for fileid, meta in self.metadata.items():
            value = meta.get(field)
            if not isinstance(value, (int, float)):
                continue
            if min_val is not None and value < min_val:
                continue
            if max_val is not None and value > max_val:
                continue
            yield fileid

    def unique_values(self, field: str) -> set[Any]:
        """Get all unique values for a metadata field.

        Args:
            field: Metadata field name.

        Returns:
            Set of unique values.

        Example:
            >>> authors = manager.unique_values("author")
            >>> print(sorted(authors))
        """
        values = set()
        for meta in self.metadata.values():
            if field in meta:
                values.add(meta[field])
        return values

    def stats(self) -> dict[str, Any]:
        """Get statistics about the metadata.

        Returns:
            Dict with counts and field coverage.
        """
        total = len(self.metadata)
        if total == 0:
            return {"total_files": 0, "fields": {}}

        field_counts: dict[str, int] = {}
        for meta in self.metadata.values():
            for field in meta:
                field_counts[field] = field_counts.get(field, 0) + 1

        return {
            "total_files": total,
            "fields": {
                field: {
                    "count": count,
                    "coverage": f"{count/total*100:.1f}%",
                }
                for field, count in sorted(field_counts.items())
            },
        }
