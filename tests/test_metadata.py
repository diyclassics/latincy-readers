"""Tests for MetadataManager."""

import json
import pytest
from pathlib import Path

from latincyreaders.utils.metadata import (
    MetadataManager,
    MetadataSchema,
    ValidationResult,
    LATIN_CORPUS_SCHEMA,
)


class TestMetadataSchema:
    """Tests for MetadataSchema validation."""

    def test_validate_required_field_present(self):
        """Required field present passes validation."""
        schema = MetadataSchema(required={"author": str})
        errors = schema.validate("test.tess", {"author": "Vergil"})
        assert errors == []

    def test_validate_required_field_missing(self):
        """Missing required field reports error."""
        schema = MetadataSchema(required={"author": str})
        errors = schema.validate("test.tess", {})
        assert len(errors) == 1
        assert "missing required field 'author'" in errors[0]

    def test_validate_required_field_wrong_type(self):
        """Wrong type for required field reports error."""
        schema = MetadataSchema(required={"date": int})
        errors = schema.validate("test.tess", {"date": "50 BCE"})
        assert len(errors) == 1
        assert "type str" in errors[0]
        assert "expected int" in errors[0]

    def test_validate_optional_field_correct_type(self):
        """Optional field with correct type passes."""
        schema = MetadataSchema(optional={"genre": str})
        errors = schema.validate("test.tess", {"genre": "epic"})
        assert errors == []

    def test_validate_optional_field_missing(self):
        """Missing optional field is not an error."""
        schema = MetadataSchema(optional={"genre": str})
        errors = schema.validate("test.tess", {})
        assert errors == []

    def test_validate_optional_field_wrong_type(self):
        """Wrong type for optional field reports error."""
        schema = MetadataSchema(optional={"date": int})
        errors = schema.validate("test.tess", {"date": "wrong"})
        assert len(errors) == 1

    def test_validate_multiple_errors(self):
        """Multiple validation errors are all reported."""
        schema = MetadataSchema(
            required={"author": str, "title": str},
            optional={"date": int},
        )
        errors = schema.validate("test.tess", {"date": "wrong"})
        assert len(errors) == 3  # 2 missing + 1 wrong type


class TestMetadataManager:
    """Tests for MetadataManager."""

    @pytest.fixture
    def metadata_dir(self, tmp_path):
        """Create a directory with metadata files."""
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        (meta_dir / "authors.json").write_text(json.dumps({
            "vergil.aen.tess": {"author": "Vergil", "title": "Aeneid"},
            "ovid.met.tess": {"author": "Ovid", "title": "Metamorphoses"},
        }))

        (meta_dir / "dates.json").write_text(json.dumps({
            "vergil.aen.tess": {"date": -19, "genre": "epic"},
            "ovid.met.tess": {"date": 8, "genre": "epic"},
            "catullus.carm.tess": {"date": -54, "genre": "lyric", "author": "Catullus"},
        }))

        return tmp_path

    def test_load_merges_metadata(self, metadata_dir):
        """MetadataManager merges data from multiple files."""
        manager = MetadataManager(metadata_dir)

        vergil = manager.get("vergil.aen.tess")
        assert vergil["author"] == "Vergil"
        assert vergil["title"] == "Aeneid"
        assert vergil["date"] == -19
        assert vergil["genre"] == "epic"

    def test_get_missing_returns_empty(self, metadata_dir):
        """get() returns empty dict for missing fileid."""
        manager = MetadataManager(metadata_dir)
        assert manager.get("nonexistent.tess") == {}

    def test_get_with_default(self, metadata_dir):
        """get() returns default for missing fileid."""
        manager = MetadataManager(metadata_dir)
        default = {"author": "Unknown"}
        assert manager.get("nonexistent.tess", default) == default

    def test_getitem_existing(self, metadata_dir):
        """Subscript access works for existing files."""
        manager = MetadataManager(metadata_dir)
        assert manager["vergil.aen.tess"]["author"] == "Vergil"

    def test_getitem_missing_raises(self, metadata_dir):
        """Subscript access raises KeyError for missing files."""
        manager = MetadataManager(metadata_dir)
        with pytest.raises(KeyError, match="nonexistent"):
            _ = manager["nonexistent.tess"]

    def test_contains(self, metadata_dir):
        """in operator works correctly."""
        manager = MetadataManager(metadata_dir)
        assert "vergil.aen.tess" in manager
        assert "nonexistent.tess" not in manager

    def test_len(self, metadata_dir):
        """len() returns count of files with metadata."""
        manager = MetadataManager(metadata_dir)
        assert len(manager) == 3

    def test_iter(self, metadata_dir):
        """Iteration yields fileids."""
        manager = MetadataManager(metadata_dir)
        fileids = list(manager)
        assert "vergil.aen.tess" in fileids
        assert len(fileids) == 3

    def test_items(self, metadata_dir):
        """items() yields (fileid, metadata) pairs."""
        manager = MetadataManager(metadata_dir)
        items = list(manager.items())
        assert len(items) == 3
        assert all(isinstance(fid, str) for fid, _ in items)
        assert all(isinstance(meta, dict) for _, meta in items)


class TestMetadataManagerValidation:
    """Tests for MetadataManager validation."""

    @pytest.fixture
    def metadata_dir(self, tmp_path):
        """Create metadata directory."""
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        return tmp_path

    def test_validate_no_schema(self, metadata_dir):
        """validate() without schema returns valid."""
        manager = MetadataManager(metadata_dir)
        result = manager.validate()
        assert result.is_valid

    def test_validate_with_schema_valid(self, metadata_dir):
        """validate() with valid data returns valid."""
        meta_dir = metadata_dir / "metadata"
        (meta_dir / "data.json").write_text(json.dumps({
            "test.tess": {"author": "Test", "date": -50},
        }))

        schema = MetadataSchema(required={"author": str})
        manager = MetadataManager(metadata_dir, schema=schema)
        result = manager.validate()
        assert result.is_valid

    def test_validate_with_schema_invalid(self, metadata_dir):
        """validate() with invalid data returns errors."""
        meta_dir = metadata_dir / "metadata"
        (meta_dir / "data.json").write_text(json.dumps({
            "test.tess": {"date": -50},  # Missing required 'author'
        }))

        schema = MetadataSchema(required={"author": str})
        manager = MetadataManager(metadata_dir, schema=schema)
        result = manager.validate()
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_validate_malformed_json(self, tmp_path):
        """validate() reports JSON parse errors."""
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        (meta_dir / "bad.json").write_text("not valid json {")

        manager = MetadataManager(tmp_path)
        # Force load to trigger the error
        _ = manager.metadata
        result = manager.validate()
        assert not result.is_valid
        assert "JSON parse error" in result.errors[0]

    def test_validation_result_bool(self):
        """ValidationResult is truthy when valid."""
        assert ValidationResult(is_valid=True)
        assert not ValidationResult(is_valid=False)


class TestMetadataManagerFiltering:
    """Tests for MetadataManager filtering methods."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with test data."""
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        (meta_dir / "data.json").write_text(json.dumps({
            "vergil.aen.tess": {"author": "Vergil", "genre": "epic", "date": -19},
            "vergil.ecl.tess": {"author": "Vergil", "genre": "pastoral", "date": -37},
            "ovid.met.tess": {"author": "Ovid", "genre": "epic", "date": 8},
            "catullus.carm.tess": {"author": "Catullus", "genre": "lyric", "date": -54},
        }))

        return MetadataManager(tmp_path)

    def test_filter_by_single_field(self, manager):
        """filter_by() with single field."""
        result = list(manager.filter_by(author="Vergil"))
        assert len(result) == 2
        assert "vergil.aen.tess" in result
        assert "vergil.ecl.tess" in result

    def test_filter_by_multiple_fields(self, manager):
        """filter_by() with multiple fields ANDs them."""
        result = list(manager.filter_by(author="Vergil", genre="epic"))
        assert result == ["vergil.aen.tess"]

    def test_filter_by_no_match(self, manager):
        """filter_by() returns empty for no matches."""
        result = list(manager.filter_by(author="Horace"))
        assert result == []

    def test_filter_by_range(self, manager):
        """filter_by_range() filters by numeric range."""
        # Before 27 BCE (date <= -27)
        # Data: vergil.aen=-19, vergil.ecl=-37, ovid.met=8, catullus=-54
        # Expected: vergil.ecl (-37) and catullus (-54)
        result = list(manager.filter_by_range("date", max_val=-27))
        assert len(result) == 2
        assert "ovid.met.tess" not in result  # 8 CE
        assert "vergil.aen.tess" not in result  # -19

    def test_filter_by_range_inclusive(self, manager):
        """filter_by_range() is inclusive on both ends."""
        result = list(manager.filter_by_range("date", -37, -19))
        assert len(result) == 2
        assert "vergil.aen.tess" in result  # -19
        assert "vergil.ecl.tess" in result  # -37

    def test_filter_by_range_no_min(self, manager):
        """filter_by_range() with no min."""
        result = list(manager.filter_by_range("date", max_val=-50))
        assert result == ["catullus.carm.tess"]

    def test_filter_by_range_no_max(self, manager):
        """filter_by_range() with no max."""
        result = list(manager.filter_by_range("date", min_val=0))
        assert result == ["ovid.met.tess"]


class TestMetadataManagerStats:
    """Tests for MetadataManager statistics."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with test data."""
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()

        (meta_dir / "data.json").write_text(json.dumps({
            "file1.tess": {"author": "A", "date": -50},
            "file2.tess": {"author": "B", "date": -25},
            "file3.tess": {"author": "A"},  # No date
        }))

        return MetadataManager(tmp_path)

    def test_unique_values(self, manager):
        """unique_values() returns all unique values for a field."""
        authors = manager.unique_values("author")
        assert authors == {"A", "B"}

    def test_unique_values_missing_field(self, manager):
        """unique_values() ignores files missing the field."""
        dates = manager.unique_values("date")
        assert dates == {-50, -25}

    def test_stats(self, manager):
        """stats() returns field coverage information."""
        stats = manager.stats()
        assert stats["total_files"] == 3
        assert "author" in stats["fields"]
        assert stats["fields"]["author"]["count"] == 3
        assert stats["fields"]["date"]["count"] == 2

    def test_stats_empty(self, tmp_path):
        """stats() handles empty metadata."""
        (tmp_path / "metadata").mkdir()
        manager = MetadataManager(tmp_path)
        stats = manager.stats()
        assert stats["total_files"] == 0
