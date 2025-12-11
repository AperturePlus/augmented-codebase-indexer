"""CLI and REPL behaviors related to path utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st

from tests.path_utils_strategies import WINDOWS_RESERVED


class TestREPLHistoryFallback:
    """Unit tests for REPL FileHistory fallback behavior."""

    def test_create_history_creates_directory(self):
        """_create_history should create parent directory if it doesn't exist."""
        from prompt_toolkit.history import FileHistory

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "subdir", "history")

            from aci.cli.repl import REPLController

            controller = object.__new__(REPLController)
            controller.console = MagicMock()

            history = controller._create_history(history_path)

            assert Path(history_path).parent.exists()
            assert isinstance(history, FileHistory)

    def test_create_history_falls_back_to_memory_on_permission_error(self):
        """_create_history should use in-memory history when directory creation fails."""
        from prompt_toolkit.history import InMemoryHistory

        from aci.cli.repl import REPLController

        controller = object.__new__(REPLController)
        controller.console = MagicMock()

        with patch("aci.cli.repl.controller.ensure_directory_exists", return_value=False):
            history = controller._create_history("/nonexistent/path/history")

        assert isinstance(history, InMemoryHistory)
        controller.console.print.assert_called()


class TestHTTPAPISystemDirectoryRejection:
    """Unit tests for HTTP API system directory rejection."""

    def test_is_system_directory_rejects_posix_paths(self):
        """is_system_directory should detect POSIX system paths."""
        from aci.core.path_utils import _is_posix_system_directory

        posix_paths = ["/etc", "/etc/passwd", "/var/log", "/usr/bin", "/bin/bash", "/proc/1", "/sys/class", "/dev/null"]

        for path in posix_paths:
            assert _is_posix_system_directory(path) is True

    def test_is_system_directory_rejects_windows_paths(self):
        """is_system_directory should detect Windows system paths."""
        from aci.core.path_utils import _is_windows_system_directory

        windows_paths = [
            (Path("C:/Windows"), "C:\\Windows"),
            (Path("C:/Windows/System32"), "C:\\Windows\\System32"),
            (Path("C:/Program Files"), "C:\\Program Files"),
            (Path("C:/Program Files (x86)"), "C:\\Program Files (x86)"),
            (Path("D:/Windows"), "D:\\Windows"),
        ]

        for mock_path, path_str in windows_paths:
            assert _is_windows_system_directory(mock_path, path_str) is True

    def test_is_system_directory_allows_user_paths(self):
        """is_system_directory should allow normal user paths."""
        from aci.core.path_utils import _is_posix_system_directory, _is_windows_system_directory

        posix_user_paths = ["/home/user/projects", "/tmp/test", "/opt/myapp"]
        for path in posix_user_paths:
            assert _is_posix_system_directory(path) is False

        windows_user_paths = [
            (Path("C:/Users/test/projects"), "C:\\Users\\test\\projects"),
            (Path("D:/Projects"), "D:\\Projects"),
            (Path("C:/MyApp"), "C:\\MyApp"),
        ]
        for mock_path, path_str in windows_user_paths:
            assert _is_windows_system_directory(mock_path, path_str) is False


class TestCodebaseContextPersistence:
    """
    **Feature: path-validation-fixes, Property 4: Codebase context persistence**
    **Validates: Requirements 4.2**
    """

    @settings(max_examples=100)
    @given(
        dirname=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).filter(lambda x: x.upper() not in WINDOWS_RESERVED)
    )
    def test_set_codebase_persists_until_changed(self, dirname: str):
        """set_codebase should persist until explicitly changed."""
        from aci.cli.repl.context import REPLContext

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            context = REPLContext()
            context.set_codebase(test_path)

            assert context.get_codebase() == test_path
            assert context.has_explicit_codebase() is True

    @settings(max_examples=100)
    @given(
        dirname1=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.upper() not in WINDOWS_RESERVED),
        dirname2=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.upper() not in WINDOWS_RESERVED),
    )
    def test_set_codebase_overwrites_previous(self, dirname1: str, dirname2: str):
        """set_codebase should overwrite previous value."""
        from aci.cli.repl.context import REPLContext

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / dirname1
            path2 = Path(tmpdir) / f"{dirname2}_second"
            path1.mkdir(exist_ok=True)
            path2.mkdir(exist_ok=True)

            context = REPLContext()

            context.set_codebase(path1)
            assert context.get_codebase() == path1

            context.set_codebase(path2)
            assert context.get_codebase() == path2

    @settings(max_examples=100)
    @given(
        dirname=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).filter(lambda x: x.upper() not in WINDOWS_RESERVED)
    )
    def test_clear_codebase_resets_to_cwd(self, dirname: str):
        """clear_codebase should reset to cwd."""
        from aci.cli.repl.context import REPLContext

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            context = REPLContext()
            context.set_codebase(test_path)
            assert context.get_codebase() == test_path

            context.clear_codebase()

            assert context.get_codebase() == Path.cwd()
            assert context.has_explicit_codebase() is False

    def test_default_codebase_is_cwd(self):
        """New REPLContext should default to cwd."""
        from aci.cli.repl.context import REPLContext

        context = REPLContext()

        assert context.get_codebase() == Path.cwd()
        assert context.has_explicit_codebase() is False


class TestUnindexedPathError:
    """
    **Feature: path-validation-fixes, Property 5: Unindexed paths produce helpful errors**
    **Validates: Requirements 4.5**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        dirname=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).filter(lambda x: x.upper() not in WINDOWS_RESERVED)
    )
    def test_unindexed_path_produces_helpful_error(self, dirname: str):
        """Use command should suggest indexing when path is unknown."""
        from aci.cli.handlers import create_use_handler
        from aci.cli.parser import ParsedCommand
        from aci.cli.repl.context import REPLContext

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            mock_store = MagicMock()
            mock_store.get_index_info.return_value = None

            context = REPLContext()
            context.set_metadata_store(mock_store)
            mock_services = MagicMock()

            handler = create_use_handler(mock_services, context)
            command = ParsedCommand(name="use", args=[str(test_path)], kwargs={})

            result = handler(command)

            assert result.success is False
            assert result.message is not None
            assert "index" in result.message.lower()
            assert "not been indexed" in result.message.lower() or "has not been indexed" in result.message

    @settings(max_examples=100, deadline=None)
    @given(
        dirname=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).filter(lambda x: x.upper() not in WINDOWS_RESERVED)
    )
    def test_indexed_path_succeeds(self, dirname: str):
        """Use command should succeed when path has been indexed."""
        from aci.cli.handlers import create_use_handler
        from aci.cli.parser import ParsedCommand
        from aci.cli.repl.context import REPLContext

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            mock_store = MagicMock()
            mock_store.get_index_info.return_value = {
                "index_id": str(test_path.resolve()),
                "root_path": str(test_path.resolve()),
            }

            context = REPLContext()
            context.set_metadata_store(mock_store)
            mock_services = MagicMock()

            handler = create_use_handler(mock_services, context)
            command = ParsedCommand(name="use", args=[str(test_path)], kwargs={})

            result = handler(command)

            assert result.success is True
            assert context.get_codebase() == test_path.resolve()

    def test_use_without_args_shows_current_codebase(self):
        """Use command without args should return current codebase info."""
        from aci.cli.handlers import create_use_handler
        from aci.cli.parser import ParsedCommand
        from aci.cli.repl.context import REPLContext

        context = REPLContext()
        mock_services = MagicMock()
        handler = create_use_handler(mock_services, context)
        command = ParsedCommand(name="use", args=[], kwargs={})

        result = handler(command)

        assert result.success is True
        assert result.message is not None
        assert "codebase" in result.message.lower() or "directory" in result.message.lower()

    def test_use_with_nonexistent_path_fails(self):
        """Use command should fail with helpful error for nonexistent path."""
        from aci.cli.handlers import create_use_handler
        from aci.cli.parser import ParsedCommand
        from aci.cli.repl.context import REPLContext

        context = REPLContext()
        mock_services = MagicMock()
        handler = create_use_handler(mock_services, context)
        command = ParsedCommand(name="use", args=["/nonexistent/path/xyz123"], kwargs={})

        result = handler(command)

        assert result.success is False
        assert result.message is not None
        assert "does not exist" in result.message

