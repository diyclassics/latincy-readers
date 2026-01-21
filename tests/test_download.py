"""Tests for DownloadableCorpusMixin."""

import pytest
from pathlib import Path

from latincyreaders.core.download import DownloadableCorpusMixin, LATINCY_DATA


class MockCorpusReader(DownloadableCorpusMixin):
    """Mock reader for testing the mixin."""

    CORPUS_URL = "https://github.com/test/test.git"
    ENV_VAR = "TEST_CORPUS_PATH"
    DEFAULT_SUBDIR = "test_corpus"
    _FILE_CHECK_PATTERN = "*.txt"


class TestDownloadableCorpusMixin:
    """Tests for the download mixin."""

    def test_default_root_uses_latincy_data(self):
        """default_root() returns path under ~/latincy_data by default."""
        root = MockCorpusReader.default_root()
        assert root == LATINCY_DATA / "test_corpus"

    def test_default_root_uses_env_var(self, monkeypatch, tmp_path):
        """default_root() respects environment variable."""
        custom_path = tmp_path / "custom"
        monkeypatch.setenv("TEST_CORPUS_PATH", str(custom_path))
        root = MockCorpusReader.default_root()
        assert root == custom_path

    def test_get_default_root_returns_existing_path(self, tmp_path, monkeypatch):
        """_get_default_root() returns path if corpus exists."""
        # Create fake corpus
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "test.txt").write_text("test")

        monkeypatch.setenv("TEST_CORPUS_PATH", str(corpus_dir))
        root = MockCorpusReader._get_default_root(auto_download=False)
        assert root == corpus_dir

    def test_get_default_root_raises_if_missing(self, tmp_path, monkeypatch):
        """_get_default_root() raises FileNotFoundError if corpus missing."""
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setenv("TEST_CORPUS_PATH", str(nonexistent))

        with pytest.raises(FileNotFoundError, match="corpus not found"):
            MockCorpusReader._get_default_root(auto_download=False)

    def test_error_message_includes_env_var(self, tmp_path, monkeypatch):
        """Error message mentions the environment variable."""
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setenv("TEST_CORPUS_PATH", str(nonexistent))

        with pytest.raises(FileNotFoundError, match="TEST_CORPUS_PATH"):
            MockCorpusReader._get_default_root(auto_download=False)

    def test_file_check_pattern_respected(self, tmp_path, monkeypatch):
        """Corpus directory must contain matching files."""
        # Create empty directory
        corpus_dir = tmp_path / "empty_corpus"
        corpus_dir.mkdir()

        monkeypatch.setenv("TEST_CORPUS_PATH", str(corpus_dir))

        # Should raise because no .txt files exist
        with pytest.raises(FileNotFoundError):
            MockCorpusReader._get_default_root(auto_download=False)

        # Now add a matching file
        (corpus_dir / "test.txt").write_text("test")
        root = MockCorpusReader._get_default_root(auto_download=False)
        assert root == corpus_dir


class TestDownloadMethod:
    """Tests for the download() class method."""

    def test_download_returns_path(self, tmp_path, monkeypatch):
        """download() returns the destination path."""
        # Skip actual git clone
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)

        dest = tmp_path / "new_corpus"
        result = MockCorpusReader.download(dest)
        assert result == dest

    def test_download_creates_parent_dirs(self, tmp_path, monkeypatch):
        """download() creates parent directories."""
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)

        deep_path = tmp_path / "a" / "b" / "c" / "corpus"
        MockCorpusReader.download(deep_path)
        assert deep_path.parent.exists()

    def test_download_skips_existing(self, tmp_path, capsys):
        """download() skips if directory already exists."""
        existing = tmp_path / "existing"
        existing.mkdir()

        result = MockCorpusReader.download(existing)
        assert result == existing

        captured = capsys.readouterr()
        assert "already exists" in captured.out
