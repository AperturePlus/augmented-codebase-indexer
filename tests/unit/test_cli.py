"""
Integration tests for CLI commands.

Tests the basic functionality of CLI commands and error handling.
"""

import re

from typer.testing import CliRunner

from aci.cli import app

runner = CliRunner()


def _plain(text: str) -> str:
    """Remove ANSI escape sequences for stable assertions."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestCLIHelp:
    """Test CLI help and usage information."""

    def test_main_help(self):
        """Main help should display available commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "index" in result.stdout
        assert "search" in result.stdout
        assert "update" in result.stdout
        assert "status" in result.stdout

    def test_index_help(self):
        """Index command help should display options."""
        result = runner.invoke(app, ["index", "--help"])

        assert result.exit_code == 0
        output = _plain(result.stdout)
        assert "Directory to index" in output
        assert "workers" in output
        assert "-w" in output

    def test_search_help(self):
        """Search command help should display options."""
        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0
        output = _plain(result.stdout)
        assert "Search query" in output
        assert "limit" in output
        assert "filter" in output

    def test_update_help(self):
        """Update command help should display options."""
        result = runner.invoke(app, ["update", "--help"])

        assert result.exit_code == 0
        assert "Directory to update" in result.stdout

    def test_status_help(self):
        """Status command help should display options."""
        result = runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0


class TestCLIErrorHandling:
    """Test CLI error handling for invalid inputs."""

    def test_index_missing_path(self):
        """Index without path should show error."""
        result = runner.invoke(app, ["index"])

        # Should fail due to missing required argument
        assert result.exit_code != 0

    def test_search_missing_query(self):
        """Search without query should show error."""
        result = runner.invoke(app, ["search"])

        # Should fail due to missing required argument
        assert result.exit_code != 0

    def test_update_missing_path(self):
        """Update without path should show error."""
        result = runner.invoke(app, ["update"])

        # Should fail due to missing required argument
        assert result.exit_code != 0

    def test_invalid_command(self):
        """Invalid command should show error."""
        result = runner.invoke(app, ["invalid_command"])

        assert result.exit_code != 0
