"""
Property-based tests for vector store defects fix.

Tests Fix 3: Server-side file path filtering
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.infrastructure.vector_store import is_glob_pattern


# =============================================================================
# Fix 3: Server-side File Path Filtering Tests
# =============================================================================


class TestGlobPatternDetection:
    """
    **Feature: search-indexing-defects-fix, Property 6 & 7: File filter detection**
    **Validates: Requirements 3.1, 3.2**

    Tests that exact paths and glob patterns are correctly distinguished.
    """

    @given(
        path=st.from_regex(r"[a-zA-Z0-9_/\-\.]+", fullmatch=True),
    )
    @settings(max_examples=100)
    def test_paths_without_wildcards_are_not_glob(self, path):
        """Paths without *, ?, [ should not be detected as glob patterns."""
        # Ensure no glob characters
        if any(c in path for c in "*?["):
            return  # Skip if hypothesis generates these

        assert not is_glob_pattern(path), f"'{path}' should not be a glob pattern"

    @given(
        base_path=st.from_regex(r"[a-zA-Z0-9_/\-]+", fullmatch=True),
        wildcard=st.sampled_from(["*", "?", "[a-z]", "*.py", "**/*.py"]),
    )
    @settings(max_examples=100)
    def test_paths_with_wildcards_are_glob(self, base_path, wildcard):
        """Paths with *, ?, [ should be detected as glob patterns."""
        glob_path = f"{base_path}/{wildcard}"
        assert is_glob_pattern(glob_path), f"'{glob_path}' should be a glob pattern"

    def test_common_exact_paths(self):
        """Common exact file paths should not be glob patterns."""
        exact_paths = [
            "src/main.py",
            "tests/test_utils.py",
            "/home/user/project/file.txt",
            "C:\\Users\\name\\file.py",
            "module/sub-module/file_name.py",
            "file.with.dots.py",
            "path/to/file-with-dashes.js",
        ]

        for path in exact_paths:
            assert not is_glob_pattern(path), f"'{path}' should not be a glob pattern"

    def test_common_glob_patterns(self):
        """Common glob patterns should be detected."""
        glob_patterns = [
            "*.py",
            "**/*.py",
            "src/*.js",
            "tests/test_*.py",
            "file?.txt",
            "[abc].py",
            "src/**/test_*.py",
            "*.{py,js}",  # Contains [ implicitly via brace expansion in some systems
        ]

        for pattern in glob_patterns:
            assert is_glob_pattern(pattern), f"'{pattern}' should be a glob pattern"


class TestExactPathServerSideFiltering:
    """
    **Feature: search-indexing-defects-fix, Property 6: Exact path uses server-side filter**
    **Validates: Requirements 3.1, 3.3**

    *For any* search with file_filter containing no glob wildcards,
    the VectorStore SHALL apply a server-side filter AND request exactly
    the specified limit.

    Note: Full integration test requires Qdrant. This tests the logic.
    """

    def test_exact_path_detection_logic(self):
        """Verify exact path detection for various inputs."""
        # These should trigger server-side filtering
        exact_paths = [
            "src/aci/services/search_service.py",
            "/absolute/path/to/file.py",
            "relative/path.txt",
        ]

        for path in exact_paths:
            is_exact = not is_glob_pattern(path)
            assert is_exact, f"'{path}' should be detected as exact path"


class TestGlobPatternClientSideFiltering:
    """
    **Feature: search-indexing-defects-fix, Property 7: Glob uses client-side filter**
    **Validates: Requirements 3.2, 3.4**

    *For any* search with file_filter containing glob wildcards,
    the VectorStore SHALL fetch at least `limit * 10` candidates AND
    apply client-side fnmatch filtering.

    Note: Full integration test requires Qdrant. This tests the logic.
    """

    def test_glob_pattern_detection_logic(self):
        """Verify glob pattern detection for various inputs."""
        # These should trigger client-side filtering with expanded fetch
        glob_patterns = [
            "*.py",
            "src/**/*.py",
            "test_?.py",
            "[a-z]*.py",
        ]

        for pattern in glob_patterns:
            is_glob = is_glob_pattern(pattern)
            assert is_glob, f"'{pattern}' should be detected as glob pattern"
