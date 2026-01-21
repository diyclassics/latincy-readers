"""Tests for DownloadableCorpusMixin."""

import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

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

    def test_download_uses_default_root_when_none(self, tmp_path, monkeypatch):
        """download(None) uses default_root()."""
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)

        fake_default = tmp_path / "default_corpus"
        with patch.object(MockCorpusReader, "default_root") as mock_default:
            mock_default.return_value = fake_default
            MockCorpusReader.download(None)
            mock_default.assert_called_once()

    def test_download_git_clone_failure(self, tmp_path, monkeypatch):
        """download() raises RuntimeError when git clone fails."""
        def mock_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "git clone")

        monkeypatch.setattr("subprocess.run", mock_run)

        dest = tmp_path / "new_corpus"
        with pytest.raises(RuntimeError, match="Failed to clone"):
            MockCorpusReader.download(dest)

    def test_download_git_not_installed(self, tmp_path, monkeypatch):
        """download() raises RuntimeError when git is not installed."""
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr("subprocess.run", mock_run)

        dest = tmp_path / "new_corpus"
        with pytest.raises(RuntimeError, match="git not found"):
            MockCorpusReader.download(dest)


class TestInteractiveDownload:
    """Tests for interactive download prompts."""

    def test_prompt_accepts_yes(self, tmp_path, monkeypatch):
        """User can accept download prompt with 'y'."""
        corpus_dir = tmp_path / "corpus"
        monkeypatch.setenv("TEST_CORPUS_PATH", str(corpus_dir))

        # Mock input to return 'y'
        monkeypatch.setattr("builtins.input", lambda _: "y")
        # Mock subprocess.run to succeed
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)

        result = MockCorpusReader._get_default_root(auto_download=True)
        assert result == corpus_dir

    def test_prompt_accepts_yes_full(self, tmp_path, monkeypatch):
        """User can accept download prompt with 'yes'."""
        corpus_dir = tmp_path / "corpus"
        monkeypatch.setenv("TEST_CORPUS_PATH", str(corpus_dir))

        monkeypatch.setattr("builtins.input", lambda _: "yes")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)

        result = MockCorpusReader._get_default_root(auto_download=True)
        assert result == corpus_dir

    def test_prompt_rejects_no(self, tmp_path, monkeypatch):
        """User can reject download prompt with 'n'."""
        corpus_dir = tmp_path / "corpus"
        monkeypatch.setenv("TEST_CORPUS_PATH", str(corpus_dir))

        monkeypatch.setattr("builtins.input", lambda _: "n")

        with pytest.raises(FileNotFoundError, match="Download manually"):
            MockCorpusReader._get_default_root(auto_download=True)

    def test_prompt_rejects_empty(self, tmp_path, monkeypatch):
        """Empty response rejects download (default is No)."""
        corpus_dir = tmp_path / "corpus"
        monkeypatch.setenv("TEST_CORPUS_PATH", str(corpus_dir))

        monkeypatch.setattr("builtins.input", lambda _: "")

        with pytest.raises(FileNotFoundError, match="Download manually"):
            MockCorpusReader._get_default_root(auto_download=True)
