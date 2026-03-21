import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aci.core.config import ACIConfig
from aci.core.path_utils import parse_runtime_path_mappings, resolve_runtime_path
from aci.mcp.context import MCPContext


def test_parse_runtime_path_mappings_supports_multiple_entries():
    mappings = parse_runtime_path_mappings(r"D:\=/host/d;/Users/alice=/host/users/alice")

    assert len(mappings) == 2
    assert mappings[0].source_prefix == "D:\\"
    assert mappings[0].target_prefix == Path("/host/d")
    assert mappings[1].source_prefix == "/Users/alice"
    assert mappings[1].target_prefix == Path("/host/users/alice")


def test_resolve_runtime_path_maps_windows_host_path_into_container(tmp_path: Path):
    mounted_repo = tmp_path / "mounted-repo"
    mounted_repo.mkdir()
    (mounted_repo / "src").mkdir()

    resolution = resolve_runtime_path(
        r"D:\projects\demo\src",
        path_mappings=parse_runtime_path_mappings(
            f"D:\\projects\\demo={mounted_repo.as_posix()}"
        ),
    )

    assert resolution.valid is True
    assert resolution.mapped is True
    assert resolution.resolved_path == (mounted_repo / "src").resolve()


def test_resolve_runtime_path_maps_posix_host_path_into_container(tmp_path: Path):
    mounted_repo = tmp_path / "mounted-repo"
    mounted_repo.mkdir()
    (mounted_repo / "src").mkdir()

    resolution = resolve_runtime_path(
        "/Users/alice/demo/src",
        path_mappings=parse_runtime_path_mappings(
            f"/Users/alice/demo={mounted_repo.as_posix()}"
        ),
    )

    assert resolution.valid is True
    assert resolution.mapped is True
    assert resolution.resolved_path == (mounted_repo / "src").resolve()


def test_resolve_runtime_path_uses_workspace_root_for_relative_path(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    target = workspace_root / "repo"
    target.mkdir(parents=True)

    resolution = resolve_runtime_path("repo", workspace_root=workspace_root)

    assert resolution.valid is True
    assert resolution.resolved_path == target.resolve()


def test_resolve_runtime_path_reports_missing_container_mapping_target(tmp_path: Path):
    resolution = resolve_runtime_path(
        r"D:\projects\missing",
        path_mappings=parse_runtime_path_mappings(
            f"D:\\projects={tmp_path.as_posix()}"
        ),
    )

    assert resolution.valid is False
    assert "not accessible inside this runtime" in resolution.error_message
    assert "ACI_MCP_PATH_MAPPINGS" in resolution.error_message


def test_mcp_index_codebase_uses_resolved_runtime_path(tmp_path: Path):
    from aci.mcp.handlers import _handle_index_codebase

    mounted_repo = tmp_path / "repo"
    mounted_repo.mkdir()

    indexing_service = MagicMock()
    indexing_service.index_directory = AsyncMock(
        return_value=MagicMock(total_files=1, total_chunks=2, duration_seconds=0.1, failed_files=[])
    )

    ctx = MCPContext(
        config=ACIConfig(),
        search_service=MagicMock(),
        indexing_service=indexing_service,
        metadata_store=MagicMock(db_path=tmp_path / "index.db"),
        vector_store=MagicMock(),
        indexing_lock=asyncio.Lock(),
        workspace_root=None,
        path_mappings=tuple(
            parse_runtime_path_mappings(f"D:\\workspace={mounted_repo.as_posix()}")
        ),
    )

    result = asyncio.run(_handle_index_codebase({"path": r"D:\workspace"}, ctx))

    assert len(result) == 1
    indexing_service.index_directory.assert_awaited_once_with(mounted_repo.resolve(), max_workers=4)
    payload = json.loads(result[0].text)
    assert payload["requested_path"] == r"D:\workspace"
    assert payload["indexed_path"] == str(mounted_repo.resolve())


def test_resolve_file_filter_pattern_keeps_wildcard_only_pattern(tmp_path: Path):
    from aci.core.path_utils import resolve_file_filter_pattern

    resolved = resolve_file_filter_pattern("**/*.tsx", tmp_path)

    assert resolved == "**/*.tsx"


def test_resolve_file_filter_pattern_expands_relative_prefixed_pattern(tmp_path: Path):
    from aci.core.path_utils import resolve_file_filter_pattern

    resolved = resolve_file_filter_pattern("apps/web/**/*.tsx", tmp_path)

    assert resolved == str(tmp_path.resolve() / "apps/web/**/*.tsx")


def test_resolve_file_filter_pattern_keeps_absolute_pattern(tmp_path: Path):
    from aci.core.path_utils import resolve_file_filter_pattern

    absolute_pattern = str(tmp_path / "apps/**/*.tsx")
    resolved = resolve_file_filter_pattern(absolute_pattern, "/another/root")

    assert resolved == absolute_pattern
