from pathlib import Path

from aci.infrastructure.codebase_registry import CodebaseRegistryStore


def test_codebase_registry_upsert_list_find_and_delete(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.db"
    store = CodebaseRegistryStore(db_path)
    try:
        root_a = tmp_path / "repo_a"
        root_b = tmp_path / "repo_a" / "nested"

        store.upsert_codebase(
            root_a,
            metadata_db_path=root_a / ".aci" / "index.db",
            collection_name="col_a",
        )
        store.upsert_codebase(
            root_b,
            metadata_db_path=root_b / ".aci" / "index.db",
            collection_name="col_b",
        )

        records = store.list_codebases()
        assert len(records) == 2

        found = store.find_codebase_for_path(root_b / "src" / "main.py")
        assert found is not None
        assert found.collection_name == "col_b"

        assert store.delete_codebase(root_b) is True
        assert store.delete_codebase(root_b) is False
    finally:
        store.close()

