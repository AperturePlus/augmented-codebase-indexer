from pathlib import Path

import pytest
import typer

import aci.cli as cli


def test_project_metadata_db_path_is_scoped_to_project_root(tmp_path: Path) -> None:
    nested = tmp_path / "repo" / "subdir"
    nested.mkdir(parents=True)

    db_path = cli._project_metadata_db_path(nested)

    assert db_path == nested.resolve() / ".aci" / "index.db"


def test_index_uses_project_scoped_metadata_db_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    indexed_path = tmp_path / "repo"
    indexed_path.mkdir()
    captured: dict[str, Path | None] = {"metadata_db_path": None}

    def fake_get_services(metadata_db_path: Path | None = None):
        captured["metadata_db_path"] = metadata_db_path
        raise RuntimeError("stop")

    monkeypatch.setattr(cli, "get_services", fake_get_services)

    with pytest.raises(typer.Exit):
        cli.index(path=indexed_path, workers=None)

    assert captured["metadata_db_path"] == indexed_path.resolve() / ".aci" / "index.db"


def test_search_uses_explicit_path_for_project_scoped_metadata_db_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    search_path = tmp_path / "repo"
    search_path.mkdir()
    captured: dict[str, Path | None] = {"metadata_db_path": None}

    def fake_get_services(metadata_db_path: Path | None = None):
        captured["metadata_db_path"] = metadata_db_path
        raise RuntimeError("stop")

    monkeypatch.setattr(cli, "get_services", fake_get_services)

    with pytest.raises(typer.Exit):
        cli.search(query="hello", path=search_path)

    assert captured["metadata_db_path"] == search_path.resolve() / ".aci" / "index.db"
