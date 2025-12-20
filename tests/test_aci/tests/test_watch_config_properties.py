"""
Property-based tests for WatchConfig.

**Feature: file-watcher-service, Property 9: Config Serialization Round-Trip**
**Feature: file-watcher-service, Property 10: Environment Variable Default**
**Validates: Requirements 7.4, 7.5**
"""

import os
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.core.watch_config import WatchConfig, _get_default_debounce_ms

# Strategies for generating valid configuration values
# Use efficient strategies instead of regex-based ones
COMMON_DIRS = ["src", "lib", "tests", "pkg", "utils", "core", "app", "data"]
COMMON_PATTERNS = ["*.pyc", "__pycache__", "*.log", ".git", "node_modules", "*.tmp"]


@st.composite
def safe_path_strategy(draw):
    """Generate a safe file path efficiently."""
    num_dirs = draw(st.integers(min_value=1, max_value=3))
    dirs = [draw(st.sampled_from(COMMON_DIRS)) for _ in range(num_dirs)]
    return "/" + "/".join(dirs)


safe_path = safe_path_strategy()

ignore_pattern = st.sampled_from(COMMON_PATTERNS)


@st.composite
def watch_config_strategy(draw):
    """Generate valid WatchConfig instances."""
    return WatchConfig(
        watch_path=Path(draw(safe_path)),
        debounce_ms=draw(st.integers(min_value=100, max_value=60000)),
        ignore_patterns=draw(st.lists(ignore_pattern, min_size=0, max_size=20)),
        verbose=draw(st.booleans()),
    )


@given(config=watch_config_strategy())
@settings(max_examples=100)
def test_watch_config_serialization_round_trip(config: WatchConfig):
    """
    **Feature: file-watcher-service, Property 9: Config Serialization Round-Trip**
    **Validates: Requirements 7.5**

    For any valid WatchConfig, serializing to dict and deserializing back
    SHALL produce an equivalent configuration.
    """
    # Serialize to dict
    config_dict = config.to_dict()

    # Deserialize from dict
    loaded_config = WatchConfig.from_dict(config_dict)

    # Verify equivalence
    assert loaded_config.watch_path == config.watch_path
    assert loaded_config.debounce_ms == config.debounce_ms
    assert loaded_config.ignore_patterns == config.ignore_patterns
    assert loaded_config.verbose == config.verbose


@given(debounce_value=st.integers(min_value=100, max_value=60000))
@settings(max_examples=100)
def test_watch_config_env_variable_default(debounce_value: int):
    """
    **Feature: file-watcher-service, Property 10: Environment Variable Default**
    **Validates: Requirements 7.4**

    For any WatchConfig created when ACI_WATCH_DEBOUNCE_MS is set,
    the debounce_ms value SHALL equal the environment variable value.
    """
    with patch.dict(os.environ, {"ACI_WATCH_DEBOUNCE_MS": str(debounce_value)}):
        # Create config without explicit debounce_ms
        config = WatchConfig(watch_path=Path("/test/path"))

        # Verify debounce_ms equals environment variable
        assert config.debounce_ms == debounce_value


def test_watch_config_default_debounce_without_env():
    """
    When ACI_WATCH_DEBOUNCE_MS is not set, debounce_ms should default to 2000ms.
    """
    with patch.dict(os.environ, {}, clear=True):
        # Remove the env var if it exists
        os.environ.pop("ACI_WATCH_DEBOUNCE_MS", None)

        # Get default value
        default_value = _get_default_debounce_ms()

        assert default_value == 2000


def test_watch_config_invalid_env_value_uses_default():
    """
    When ACI_WATCH_DEBOUNCE_MS contains an invalid value, use default 2000ms.
    """
    with patch.dict(os.environ, {"ACI_WATCH_DEBOUNCE_MS": "not_a_number"}):
        default_value = _get_default_debounce_ms()
        assert default_value == 2000


def test_watch_config_string_path_conversion():
    """
    WatchConfig should convert string paths to Path objects.
    """
    config = WatchConfig(watch_path="/test/path", debounce_ms=1000)  # type: ignore
    assert isinstance(config.watch_path, Path)
    assert config.watch_path == Path("/test/path")


def test_watch_config_to_dict_types():
    """
    to_dict() should return JSON-serializable types.
    """
    config = WatchConfig(
        watch_path=Path("/test/path"),
        debounce_ms=2000,
        ignore_patterns=["*.pyc", "__pycache__"],
        verbose=True,
    )

    config_dict = config.to_dict()

    # Verify types are JSON-serializable
    assert isinstance(config_dict["watch_path"], str)
    assert isinstance(config_dict["debounce_ms"], int)
    assert isinstance(config_dict["ignore_patterns"], list)
    assert isinstance(config_dict["verbose"], bool)
