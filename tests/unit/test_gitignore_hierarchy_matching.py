"""Regression tests for gitignore hierarchy matching behavior."""

from pathlib import Path

from aci.core.gitignore_manager import GitignoreManager


def test_nested_directory_pattern_ignores_files_under_that_directory(tmp_path):
    """A nested .gitignore pattern like `out/` should ignore files inside that directory."""
    (tmp_path / "apps" / "web").mkdir(parents=True)
    (tmp_path / "apps" / "web" / ".gitignore").write_text("out/\n", encoding="utf-8")

    manager = GitignoreManager(tmp_path)
    manager.load_gitignore(tmp_path / "apps" / "web" / ".gitignore")

    assert manager.matches(Path("apps") / "web" / "out", is_dir=True)
    assert manager.matches(Path("apps") / "web" / "out" / "_next" / "static" / "chunks" / "framework.js", is_dir=False)
    assert not manager.matches(Path("apps") / "web" / "src" / "app.js", is_dir=False)


def test_nested_anchored_pattern_is_scoped_to_its_gitignore_directory(tmp_path):
    """Anchored patterns in nested .gitignore files are anchored to that directory, not repository root."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / ".gitignore").write_text("/build\n", encoding="utf-8")

    manager = GitignoreManager(tmp_path)
    manager.load_gitignore(tmp_path / "a" / ".gitignore")

    assert manager.matches(Path("a") / "build", is_dir=False)
    assert not manager.matches(Path("build"), is_dir=False)
    assert not manager.matches(Path("a") / "b" / "build", is_dir=False)


def test_nested_slash_pattern_is_scoped_to_its_gitignore_directory(tmp_path):
    """Patterns containing slashes in nested .gitignore should stay scoped to that subtree."""
    (tmp_path / "sub" / "nested" / "tmp").mkdir(parents=True)
    (tmp_path / "other" / "tmp").mkdir(parents=True)
    (tmp_path / "sub" / ".gitignore").write_text("tmp/*.js\n", encoding="utf-8")

    manager = GitignoreManager(tmp_path)
    manager.load_gitignore(tmp_path / "sub" / ".gitignore")

    assert manager.matches(Path("sub") / "tmp" / "file.js", is_dir=False)
    assert not manager.matches(Path("sub") / "nested" / "tmp" / "file.js", is_dir=False)
    assert not manager.matches(Path("other") / "tmp" / "file.js", is_dir=False)
