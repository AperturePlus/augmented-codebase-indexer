"""
Property-based tests for history manager.

**Feature: interactive-repl, Property 4: History persistence round trip**
**Validates: Requirements 4.3, 4.4**
"""

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.cli.history import HistoryManager

# Strategy for valid command strings (non-empty, no newlines)
# Commands should be printable ASCII without newlines
command_string = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Zs"),
        blacklist_characters="\n\r",
    ),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip())


# Strategy for lists of commands
command_list = st.lists(command_string, min_size=0, max_size=50)


@given(commands=command_list)
@settings(max_examples=100)
def test_history_persistence_round_trip(commands: list[str]):
    """
    **Feature: interactive-repl, Property 4: History persistence round trip**
    **Validates: Requirements 4.3, 4.4**

    For any list of non-empty command strings, saving to history file
    then loading should return the same list of commands in the same order.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / ".aci" / "history"
        manager = HistoryManager(history_file)

        # Save the commands
        manager.save(commands)

        # Create a new manager instance to simulate new session
        new_manager = HistoryManager(history_file)

        # Load the commands
        loaded = new_manager.load()

        # Filter expected commands (save filters empty ones)
        expected = [cmd for cmd in commands if cmd.strip()]

        # Verify round trip
        assert loaded == expected


@given(commands=command_list)
@settings(max_examples=100)
def test_history_append_accumulates(commands: list[str]):
    """
    **Feature: interactive-repl, Property 4: History persistence round trip**
    **Validates: Requirements 4.3, 4.4**

    For any sequence of commands appended one by one, loading should
    return all commands in the order they were appended.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / ".aci" / "history"
        manager = HistoryManager(history_file)

        # Append commands one by one
        for cmd in commands:
            manager.append(cmd)

        # Create new manager to simulate new session
        new_manager = HistoryManager(history_file)
        loaded = new_manager.load()

        # Filter expected (append filters empty ones)
        expected = [cmd for cmd in commands if cmd.strip()]

        assert loaded == expected


def test_load_nonexistent_file_returns_empty():
    """
    **Feature: interactive-repl, Property 4: History persistence round trip**
    **Validates: Requirements 4.4**

    Loading from a nonexistent file should return an empty list
    without raising an error (graceful handling on first run).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "nonexistent" / "history"
        manager = HistoryManager(history_file)

        loaded = manager.load()

        assert loaded == []


def test_empty_commands_filtered():
    """
    **Feature: interactive-repl, Property 4: History persistence round trip**
    **Validates: Requirements 4.3, 4.4**

    Empty or whitespace-only commands should be filtered out
    when saving and appending.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / ".aci" / "history"
        manager = HistoryManager(history_file)

        # Save with empty commands mixed in
        manager.save(["cmd1", "", "cmd2", "   ", "cmd3"])

        new_manager = HistoryManager(history_file)
        loaded = new_manager.load()

        assert loaded == ["cmd1", "cmd2", "cmd3"]


def test_append_empty_command_ignored():
    """
    **Feature: interactive-repl, Property 4: History persistence round trip**
    **Validates: Requirements 4.3, 4.4**

    Appending empty or whitespace-only commands should be ignored.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / ".aci" / "history"
        manager = HistoryManager(history_file)

        manager.append("cmd1")
        manager.append("")
        manager.append("   ")
        manager.append("cmd2")

        new_manager = HistoryManager(history_file)
        loaded = new_manager.load()

        assert loaded == ["cmd1", "cmd2"]


def test_get_history_returns_copy():
    """Test that get_history returns a copy, not the internal list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / ".aci" / "history"
        manager = HistoryManager(history_file)

        manager.append("cmd1")
        history = manager.get_history()
        history.append("modified")

        assert manager.get_history() == ["cmd1"]


def test_clear_removes_history():
    """Test that clear removes all history from memory and disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / ".aci" / "history"
        manager = HistoryManager(history_file)

        manager.save(["cmd1", "cmd2"])
        manager.clear()

        assert manager.get_history() == []
        assert not history_file.exists()
