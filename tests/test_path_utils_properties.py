"""
Property-based tests for path validation utilities.

**Feature: path-validation-fixes**

Uses Hypothesis for property-based testing with minimum 100 iterations.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st

from aci.core.path_utils import (
    PathValidationResult,
    is_system_directory,
    validate_indexable_path,
    ensure_directory_exists,
    POSIX_SYSTEM_DIRS,
    WINDOWS_SYSTEM_DIRS,
)


# Strategies for generating test data
@st.composite
def non_existent_paths(draw):
    """Generate paths that are very unlikely to exist."""
    # Generate random path components
    components = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=10
        ),
        min_size=2,
        max_size=5
    ))
    # Add a unique suffix to ensure non-existence
    suffix = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=8,
        max_size=16
    ))
    return os.path.join(tempfile.gettempdir(), *components, f"nonexistent_{suffix}")


@st.composite
def posix_system_paths(draw):
    """Generate paths under POSIX system directories."""
    base = draw(st.sampled_from(list(POSIX_SYSTEM_DIRS)))
    subpath = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P'), min_codepoint=45, max_codepoint=122),
        min_size=0,
        max_size=20
    ).filter(lambda x: '/' not in x and '\\' not in x))
    if subpath:
        return f"{base}/{subpath}"
    return base


@st.composite  
def windows_system_paths(draw):
    """Generate paths under Windows system directories."""
    drive = draw(st.sampled_from(['C:', 'D:']))
    sys_dir = draw(st.sampled_from(list(WINDOWS_SYSTEM_DIRS)))
    # Capitalize properly for Windows
    sys_dir_proper = sys_dir.title() if sys_dir != "program files (x86)" else "Program Files (x86)"
    subpath = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=0,
        max_size=20
    ))
    if subpath:
        return f"{drive}\\{sys_dir_proper}\\{subpath}"
    return f"{drive}\\{sys_dir_proper}"


class TestPathExistenceValidation:
    """
    **Feature: path-validation-fixes, Property 1: Non-existent paths are rejected**
    **Validates: Requirements 2.1, 2.3**
    """

    @settings(max_examples=100)
    @given(path=non_existent_paths())
    def test_non_existent_paths_rejected(self, path: str):
        """
        Property 1: For any path that does not exist, validate_indexable_path()
        SHALL return valid=False with error message containing 'does not exist'.
        """
        # Ensure path doesn't actually exist (very unlikely but check)
        if Path(path).exists():
            pytest.skip("Generated path unexpectedly exists")
        
        result = validate_indexable_path(path)
        
        assert result.valid is False
        assert result.error_message is not None
        assert "does not exist" in result.error_message


class TestDirectoryTypeValidation:
    """
    **Feature: path-validation-fixes, Property 2: Non-directory paths are rejected**
    **Validates: Requirements 2.2, 2.4**
    """

    @settings(max_examples=100)
    @given(filename=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=20
    ))
    def test_file_paths_rejected(self, filename: str):
        """
        Property 2: For any path that exists but is not a directory,
        validate_indexable_path() SHALL return valid=False with error
        message containing 'not a directory'.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file (not a directory)
            file_path = Path(tmpdir) / f"file_{filename}.txt"
            file_path.write_text("test content")
            
            result = validate_indexable_path(file_path)
            
            assert result.valid is False
            assert result.error_message is not None
            assert "not a directory" in result.error_message


class TestSystemDirectoryDetection:
    """
    **Feature: path-validation-fixes, Property 3: System directories are detected across platforms**
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    """

    @settings(max_examples=100)
    @given(path=posix_system_paths())
    def test_posix_system_directories_detected(self, path: str):
        """
        Property 3a: For any path under POSIX system directories,
        is_system_directory() SHALL return True when on POSIX.
        """
        # Test the internal function directly to avoid path resolution issues on Windows
        from aci.core.path_utils import _is_posix_system_directory
        result = _is_posix_system_directory(path)
        assert result is True, f"Expected {path} to be detected as POSIX system directory"

    @settings(max_examples=100)
    @given(path=windows_system_paths())
    def test_windows_system_directories_detected(self, path: str):
        """
        Property 3b: For any path under Windows system directories,
        is_system_directory() SHALL return True when on Windows.
        """
        # Test the internal function directly with a mock Path object
        from aci.core.path_utils import _is_windows_system_directory
        mock_path = Path(path)
        result = _is_windows_system_directory(mock_path, path)
        assert result is True, f"Expected {path} to be detected as Windows system directory"

    @settings(max_examples=100)
    @given(dirname=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=5,
        max_size=20
    ).filter(lambda x: x.lower() not in WINDOWS_SYSTEM_DIRS))
    def test_non_system_directories_not_flagged(self, dirname: str):
        """
        Property 3c: For any path NOT under system directories,
        is_system_directory() SHALL return False.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            
            result = is_system_directory(test_path)
            assert result is False, f"Expected {test_path} to NOT be detected as system directory"


class TestValidDirectoryAccepted:
    """Test that valid directories pass validation."""

    # Windows reserved device names that can't be used as directory names
    WINDOWS_RESERVED = frozenset([
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
    ])

    @settings(max_examples=100)
    @given(dirname=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.upper() not in TestValidDirectoryAccepted.WINDOWS_RESERVED))
    def test_valid_directories_accepted(self, dirname: str):
        """Valid directories that exist and are not system dirs should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            
            result = validate_indexable_path(test_path)
            
            assert result.valid is True
            assert result.error_message is None


class TestEnsureDirectoryExists:
    """Test ensure_directory_exists function."""

    # Windows reserved device names that can't be used as directory names
    WINDOWS_RESERVED = frozenset([
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
    ])

    @settings(max_examples=50)
    @given(parts=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=10
        ).filter(lambda x: x.upper() not in TestEnsureDirectoryExists.WINDOWS_RESERVED),
        min_size=1,
        max_size=3
    ))
    def test_creates_nested_directories(self, parts: list):
        """ensure_directory_exists creates nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir).joinpath(*parts)
            
            result = ensure_directory_exists(nested_path)
            
            assert result is True
            assert nested_path.exists()
            assert nested_path.is_dir()



class TestREPLHistoryFallback:
    """
    Unit tests for REPL FileHistory fallback behavior.
    **Validates: Requirements 1.3**
    """

    def test_create_history_creates_directory(self):
        """_create_history should create parent directory if it doesn't exist."""
        from unittest.mock import MagicMock
        from prompt_toolkit.history import FileHistory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "subdir", "history")
            
            # Create a minimal mock controller to test _create_history
            from aci.cli.repl import REPLController
            
            # Create instance without full initialization
            controller = object.__new__(REPLController)
            controller.console = MagicMock()
            
            # Call _create_history directly
            history = controller._create_history(history_path)
            
            # Verify directory was created
            assert Path(history_path).parent.exists()
            # Verify FileHistory was returned
            assert isinstance(history, FileHistory)

    def test_create_history_falls_back_to_memory_on_permission_error(self):
        """_create_history should use in-memory history when directory creation fails."""
        from unittest.mock import MagicMock, patch
        from prompt_toolkit.history import InMemoryHistory
        
        from aci.cli.repl import REPLController
        
        # Create instance without full initialization
        controller = object.__new__(REPLController)
        controller.console = MagicMock()
        
        # Mock ensure_directory_exists to return False (simulating permission error)
        with patch('aci.cli.repl.controller.ensure_directory_exists', return_value=False):
            history = controller._create_history("/nonexistent/path/history")
        
        # Verify in-memory history is used
        assert isinstance(history, InMemoryHistory)
        # Verify warning was printed
        controller.console.print.assert_called()



class TestHTTPAPISystemDirectoryRejection:
    """
    Unit tests for HTTP API system directory rejection.
    **Validates: Requirements 3.1, 3.2**
    """

    def test_is_system_directory_rejects_posix_paths(self):
        """is_system_directory should detect POSIX system paths."""
        from aci.core.path_utils import _is_posix_system_directory
        
        posix_paths = [
            "/etc",
            "/etc/passwd",
            "/var/log",
            "/usr/bin",
            "/bin/bash",
            "/proc/1",
            "/sys/class",
            "/dev/null",
        ]
        
        for path in posix_paths:
            assert _is_posix_system_directory(path) is True, f"Expected {path} to be system dir"

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
            result = _is_windows_system_directory(mock_path, path_str)
            assert result is True, f"Expected {path_str} to be system dir"

    def test_is_system_directory_allows_user_paths(self):
        """is_system_directory should allow normal user paths."""
        from aci.core.path_utils import _is_posix_system_directory, _is_windows_system_directory
        
        # POSIX user paths
        posix_user_paths = [
            "/home/user/projects",
            "/tmp/test",
            "/opt/myapp",
        ]
        for path in posix_user_paths:
            assert _is_posix_system_directory(path) is False, f"Expected {path} to NOT be system dir"
        
        # Windows user paths
        windows_user_paths = [
            (Path("C:/Users/test/projects"), "C:\\Users\\test\\projects"),
            (Path("D:/Projects"), "D:\\Projects"),
            (Path("C:/MyApp"), "C:\\MyApp"),
        ]
        for mock_path, path_str in windows_user_paths:
            result = _is_windows_system_directory(mock_path, path_str)
            assert result is False, f"Expected {path_str} to NOT be system dir"


class TestCodebaseContextPersistence:
    """
    **Feature: path-validation-fixes, Property 4: Codebase context persistence**
    **Validates: Requirements 4.2**
    """

    # Windows reserved device names that can't be used as directory names
    WINDOWS_RESERVED = frozenset([
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
    ])

    @settings(max_examples=100)
    @given(dirname=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.upper() not in TestCodebaseContextPersistence.WINDOWS_RESERVED))
    def test_set_codebase_persists_until_changed(self, dirname: str):
        """
        Property 4: For any valid directory path, after calling set_codebase(path),
        subsequent calls to get_codebase() SHALL return that same path until
        set_codebase() or clear_codebase() is called.
        """
        from aci.cli.repl.context import REPLContext
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            
            context = REPLContext()
            
            # Set the codebase
            context.set_codebase(test_path)
            
            # Multiple calls to get_codebase should return the same path
            assert context.get_codebase() == test_path
            assert context.get_codebase() == test_path
            assert context.get_codebase() == test_path
            
            # has_explicit_codebase should return True
            assert context.has_explicit_codebase() is True

    @settings(max_examples=100)
    @given(
        dirname1=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=10
        ).filter(lambda x: x.upper() not in TestCodebaseContextPersistence.WINDOWS_RESERVED),
        dirname2=st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=10
        ).filter(lambda x: x.upper() not in TestCodebaseContextPersistence.WINDOWS_RESERVED)
    )
    def test_set_codebase_overwrites_previous(self, dirname1: str, dirname2: str):
        """
        Property 4b: Setting a new codebase should overwrite the previous one.
        """
        from aci.cli.repl.context import REPLContext
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / dirname1
            path2 = Path(tmpdir) / f"{dirname2}_second"  # Ensure different from path1
            path1.mkdir(exist_ok=True)
            path2.mkdir(exist_ok=True)
            
            context = REPLContext()
            
            # Set first codebase
            context.set_codebase(path1)
            assert context.get_codebase() == path1
            
            # Set second codebase - should overwrite
            context.set_codebase(path2)
            assert context.get_codebase() == path2

    @settings(max_examples=100)
    @given(dirname=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.upper() not in TestCodebaseContextPersistence.WINDOWS_RESERVED))
    def test_clear_codebase_resets_to_cwd(self, dirname: str):
        """
        Property 4c: After clear_codebase(), get_codebase() should return cwd.
        """
        from aci.cli.repl.context import REPLContext
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            
            context = REPLContext()
            
            # Set and then clear
            context.set_codebase(test_path)
            assert context.get_codebase() == test_path
            
            context.clear_codebase()
            
            # Should now return cwd
            assert context.get_codebase() == Path.cwd()
            assert context.has_explicit_codebase() is False

    def test_default_codebase_is_cwd(self):
        """New context should default to current working directory."""
        from aci.cli.repl.context import REPLContext
        
        context = REPLContext()
        
        assert context.get_codebase() == Path.cwd()
        assert context.has_explicit_codebase() is False


class TestUnindexedPathError:
    """
    **Feature: path-validation-fixes, Property 5: Unindexed paths produce helpful errors**
    **Validates: Requirements 4.5**
    """

    # Windows reserved device names that can't be used as directory names
    WINDOWS_RESERVED = frozenset([
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
    ])

    @settings(max_examples=100, deadline=None)
    @given(dirname=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.upper() not in TestUnindexedPathError.WINDOWS_RESERVED))
    def test_unindexed_path_produces_helpful_error(self, dirname: str):
        """
        Property 5: For any path that exists but has not been indexed,
        the use command SHALL return an error message suggesting to run
        the index command.
        """
        from unittest.mock import MagicMock
        from aci.cli.repl.context import REPLContext
        from aci.cli.handlers import create_use_handler
        from aci.cli.parser import ParsedCommand
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            
            # Create a mock metadata store that returns None (not indexed)
            mock_store = MagicMock()
            mock_store.get_index_info.return_value = None
            
            # Create context with mock store
            context = REPLContext()
            context.set_metadata_store(mock_store)
            
            # Create mock services
            mock_services = MagicMock()
            
            # Create the use handler
            handler = create_use_handler(mock_services, context)
            
            # Create a command with the test path
            command = ParsedCommand(name="use", args=[str(test_path)], kwargs={})
            
            # Execute the handler
            result = handler(command)
            
            # Verify the result indicates failure
            assert result.success is False
            assert result.message is not None
            # Verify the error message suggests running index
            assert "index" in result.message.lower()
            assert "not been indexed" in result.message.lower() or "has not been indexed" in result.message

    @settings(max_examples=100, deadline=None)
    @given(dirname=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.upper() not in TestUnindexedPathError.WINDOWS_RESERVED))
    def test_indexed_path_succeeds(self, dirname: str):
        """
        Property 5b: For any path that exists AND has been indexed,
        the use command SHALL succeed and set the codebase.
        """
        from unittest.mock import MagicMock
        from aci.cli.repl.context import REPLContext
        from aci.cli.handlers import create_use_handler
        from aci.cli.parser import ParsedCommand
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            
            # Create a mock metadata store that returns index info (indexed)
            mock_store = MagicMock()
            mock_store.get_index_info.return_value = {
                "index_id": str(test_path.resolve()),
                "root_path": str(test_path.resolve()),
            }
            
            # Create context with mock store
            context = REPLContext()
            context.set_metadata_store(mock_store)
            
            # Create mock services
            mock_services = MagicMock()
            
            # Create the use handler
            handler = create_use_handler(mock_services, context)
            
            # Create a command with the test path
            command = ParsedCommand(name="use", args=[str(test_path)], kwargs={})
            
            # Execute the handler
            result = handler(command)
            
            # Verify the result indicates success
            assert result.success is True
            # Verify the codebase was set
            assert context.get_codebase() == test_path.resolve()

    def test_use_without_args_shows_current_codebase(self):
        """
        Use command without arguments should display current codebase.
        """
        from unittest.mock import MagicMock
        from aci.cli.repl.context import REPLContext
        from aci.cli.handlers import create_use_handler
        from aci.cli.parser import ParsedCommand
        
        # Create context
        context = REPLContext()
        
        # Create mock services
        mock_services = MagicMock()
        
        # Create the use handler
        handler = create_use_handler(mock_services, context)
        
        # Create a command without args
        command = ParsedCommand(name="use", args=[], kwargs={})
        
        # Execute the handler
        result = handler(command)
        
        # Verify the result indicates success
        assert result.success is True
        # Verify the message contains path info
        assert result.message is not None
        assert "codebase" in result.message.lower() or "directory" in result.message.lower()

    def test_use_with_nonexistent_path_fails(self):
        """
        Use command with non-existent path should fail with helpful error.
        """
        from unittest.mock import MagicMock
        from aci.cli.repl.context import REPLContext
        from aci.cli.handlers import create_use_handler
        from aci.cli.parser import ParsedCommand
        
        # Create context
        context = REPLContext()
        
        # Create mock services
        mock_services = MagicMock()
        
        # Create the use handler
        handler = create_use_handler(mock_services, context)
        
        # Create a command with non-existent path
        command = ParsedCommand(name="use", args=["/nonexistent/path/xyz123"], kwargs={})
        
        # Execute the handler
        result = handler(command)
        
        # Verify the result indicates failure
        assert result.success is False
        assert result.message is not None
        assert "does not exist" in result.message
