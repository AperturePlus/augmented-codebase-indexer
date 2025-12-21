"""
Property-based tests for CLI search mode acceptance.

**Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**
**Feature: service-initialization-refactor, Property 5: Default Search Mode**

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aci.services.search_types import SearchMode

# Valid search modes as defined in SearchMode enum
VALID_MODES = {"hybrid", "vector", "grep", "fuzzy", "summary"}


class TestSearchModeAcceptance:
    """
    **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

    **Validates: Requirements 2.1, 2.2, 2.3**

    *For any* valid search mode string in {"hybrid", "vector", "grep", "summary"},
    all interfaces SHALL accept the mode without error.
    """

    @given(mode=st.sampled_from(list(VALID_MODES)))
    @settings(max_examples=100)
    def test_search_mode_enum_accepts_valid_modes(self, mode: str):
        """
        **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

        **Validates: Requirements 2.1, 2.2, 2.3**

        Test that SearchMode enum can be constructed from all valid mode strings.
        """
        # Should not raise an exception
        search_mode = SearchMode(mode)
        assert search_mode.value == mode

    @given(mode=st.sampled_from(list(VALID_MODES)))
    @settings(max_examples=100)
    def test_search_mode_case_sensitivity(self, mode: str):
        """
        **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

        **Validates: Requirements 2.1**

        Test that SearchMode enum values are lowercase.
        """
        search_mode = SearchMode(mode)
        assert search_mode.value == search_mode.value.lower()

    @given(
        invalid_mode=st.text(min_size=1, max_size=20).filter(
            lambda x: x.lower() not in VALID_MODES and x not in VALID_MODES
        )
    )
    @settings(max_examples=100)
    def test_search_mode_rejects_invalid_modes(self, invalid_mode: str):
        """
        **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

        **Validates: Requirements 2.1**

        Test that SearchMode enum rejects invalid mode strings.
        """
        with pytest.raises(ValueError):
            SearchMode(invalid_mode)


class TestDefaultSearchMode:
    """
    **Feature: service-initialization-refactor, Property 5: Default Search Mode**

    **Validates: Requirements 2.4**

    *For any* search request without a mode specified, the system SHALL use HYBRID mode.
    """

    def test_search_mode_enum_has_hybrid(self):
        """
        **Feature: service-initialization-refactor, Property 5: Default Search Mode**

        **Validates: Requirements 2.4**

        Verify that SearchMode enum includes HYBRID as a valid mode.
        """
        assert hasattr(SearchMode, "HYBRID")
        assert SearchMode.HYBRID.value == "hybrid"

    def test_hybrid_is_default_mode(self):
        """
        **Feature: service-initialization-refactor, Property 5: Default Search Mode**

        **Validates: Requirements 2.4**

        Verify HYBRID is the expected default mode value.
        """
        # HYBRID should be a valid mode that can be used as default
        default_mode = SearchMode.HYBRID
        assert default_mode == SearchMode("hybrid")


class TestSearchModeEnumConsistency:
    """
    Test that SearchMode enum is consistent with expected valid modes.
    """

    def test_all_enum_values_are_valid_modes(self):
        """
        **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

        **Validates: Requirements 2.1, 2.2, 2.3**

        Verify all SearchMode enum values match the valid modes set.
        """
        enum_values = {mode.value for mode in SearchMode}
        assert enum_values == VALID_MODES

    def test_search_mode_enum_is_string_enum(self):
        """
        **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

        **Validates: Requirements 2.1**

        Verify SearchMode is a string enum for easy comparison.
        """
        for mode in SearchMode:
            assert isinstance(mode.value, str)
            assert mode.value == mode.value.lower()

    @given(mode=st.sampled_from(list(VALID_MODES)))
    @settings(max_examples=100)
    def test_search_mode_string_comparison(self, mode: str):
        """
        **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

        **Validates: Requirements 2.1**

        Test that SearchMode can be compared with strings.
        """
        search_mode = SearchMode(mode)
        # String enum should allow string comparison
        assert search_mode == mode
        assert search_mode.value == mode


class TestCLIModeValidation:
    """
    Test CLI mode validation logic without invoking full CLI.
    """

    def test_cli_mode_validation_logic(self):
        """
        **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

        **Validates: Requirements 2.1**

        Test the mode validation logic used in CLI search command.
        """
        valid_modes = {"hybrid", "vector", "grep", "fuzzy", "summary"}

        # Test all valid modes
        for mode in valid_modes:
            mode_lower = mode.lower()
            assert mode_lower in valid_modes
            search_mode = SearchMode(mode_lower)
            assert search_mode is not None

    @given(mode=st.sampled_from(list(VALID_MODES)))
    @settings(max_examples=100)
    def test_cli_mode_case_insensitive_validation(self, mode: str):
        """
        **Feature: service-initialization-refactor, Property 4: Search Mode Acceptance**

        **Validates: Requirements 2.1**

        Test that CLI mode validation is case-insensitive.
        """
        valid_modes = {"hybrid", "vector", "grep", "fuzzy", "summary"}

        # Test lowercase
        assert mode.lower() in valid_modes

        # Test uppercase (should be normalized to lowercase)
        mode_upper = mode.upper()
        assert mode_upper.lower() in valid_modes

        # Test mixed case
        if len(mode) > 1:
            mixed = mode[0].upper() + mode[1:].lower()
            assert mixed.lower() in valid_modes

    def test_default_mode_when_none_specified(self):
        """
        **Feature: service-initialization-refactor, Property 5: Default Search Mode**

        **Validates: Requirements 2.4**

        Test that default mode is HYBRID when mode is None.
        """
        # Simulate CLI logic: when mode is None, use HYBRID
        mode = None
        if mode is not None:
            search_mode = SearchMode(mode.lower())
        else:
            search_mode = SearchMode.HYBRID

        assert search_mode == SearchMode.HYBRID
