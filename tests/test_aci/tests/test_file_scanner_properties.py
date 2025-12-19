"""
Property-based tests for FileScanner.

**Feature: codebase-semantic-search, Property 1: File Extension Filter Correctness**
**Feature: codebase-semantic-search, Property 3: Ignore Pattern Exclusion**
**Validates: Requirements 1.1, 1.4**
"""

import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.core.file_scanner import FileScanner

# Strategies for generating valid test data

# File extension strategy - generates extensions like .py, .js, .txt
file_extension = st.from_regex(r"\.[a-z]{1,5}", fullmatch=True)

# Safe filename strategy - alphanumeric with underscores
safe_filename = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)

# Directory name strategy
dir_name = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)


@st.composite
def file_structure_strategy(draw):
    """
    Generate a random file structure with files having various extensions.

    Returns:
        tuple: (list of (relative_path, extension) tuples, set of extensions to filter)
    """
    # Generate 1-10 unique extensions
    all_extensions = draw(st.lists(file_extension, min_size=2, max_size=10, unique=True))
    assume(len(all_extensions) >= 2)

    # Select a subset of extensions to filter for (at least 1)
    num_filter_extensions = draw(st.integers(min_value=1, max_value=len(all_extensions)))
    filter_extensions = set(all_extensions[:num_filter_extensions])

    # Generate files with random extensions
    files = []
    num_files = draw(st.integers(min_value=1, max_value=20))

    for i in range(num_files):
        # Randomly choose depth (0-2 subdirectories)
        depth = draw(st.integers(min_value=0, max_value=2))

        # Build path components
        path_parts = []
        for _ in range(depth):
            path_parts.append(draw(dir_name))

        # Add filename with random extension from all_extensions
        ext = draw(st.sampled_from(all_extensions))
        filename = draw(safe_filename) + ext
        path_parts.append(filename)

        rel_path = "/".join(path_parts)
        files.append((rel_path, ext))

    return files, filter_extensions


def create_test_directory(tmpdir: Path, files: list[tuple[str, str]]) -> None:
    """
    Create a test directory structure with the given files.

    Args:
        tmpdir: Temporary directory path
        files: List of (relative_path, extension) tuples
    """
    for rel_path, _ in files:
        file_path = tmpdir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"# Content of {rel_path}\n", encoding="utf-8")


@given(data=file_structure_strategy())
@settings(max_examples=100, deadline=None)
def test_file_extension_filter_correctness(data):
    """
    **Feature: codebase-semantic-search, Property 1: File Extension Filter Correctness**
    **Validates: Requirements 1.1**

    For any directory structure and any configured file extension set,
    FileScanner returns only files whose extensions belong to the configured set.
    """
    files, filter_extensions = data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create the test directory structure
        create_test_directory(tmpdir_path, files)

        # Create scanner with the filter extensions
        scanner = FileScanner(
            extensions=filter_extensions,
            ignore_patterns=[],  # No ignore patterns for this test
        )

        # Scan the directory
        scanned_files = list(scanner.scan(tmpdir_path))

        # Property: All returned files must have extensions in the filter set
        for scanned_file in scanned_files:
            ext = scanned_file.path.suffix.lower()
            assert ext in filter_extensions, (
                f"File {scanned_file.path} has extension {ext} "
                f"which is not in filter set {filter_extensions}"
            )

        # Additional check: All files with matching extensions should be returned
        expected_files = {
            (tmpdir_path / rel_path).resolve()
            for rel_path, ext in files
            if ext in filter_extensions
        }
        actual_files = {sf.path for sf in scanned_files}

        assert actual_files == expected_files, (
            f"Expected files: {expected_files}, Actual files: {actual_files}"
        )


@st.composite
def ignore_pattern_strategy(draw):
    """
    Generate a test scenario with files and ignore patterns.

    Returns:
        tuple: (list of (relative_path, should_be_ignored) tuples,
                list of ignore patterns,
                set of extensions)
    """
    # Use a fixed extension set for simplicity
    extensions = {".py", ".js", ".txt"}

    # Generate ignore patterns - use simple directory/file patterns
    # Pattern types: exact directory name, wildcard extension, exact filename
    pattern_types = draw(
        st.lists(st.sampled_from(["dir", "ext", "file"]), min_size=1, max_size=3, unique=True)
    )

    patterns = []
    ignored_dirs = set()
    ignored_extensions = set()
    ignored_files = set()

    for ptype in pattern_types:
        if ptype == "dir":
            # Directory pattern like "node_modules" or "__pycache__"
            dirname = draw(st.sampled_from(["ignored_dir", "skip_this", "temp"]))
            patterns.append(dirname)
            ignored_dirs.add(dirname)
        elif ptype == "ext":
            # Extension pattern like "*.log"
            ext = draw(st.sampled_from([".log", ".tmp", ".bak"]))
            patterns.append(f"*{ext}")
            ignored_extensions.add(ext)
        else:  # file
            # Specific filename pattern
            filename = draw(st.sampled_from(["ignore_me.py", "skip.js", "temp.txt"]))
            patterns.append(filename)
            ignored_files.add(filename)

    # Generate files - mix of ignored and non-ignored
    files = []

    # Add some files that should NOT be ignored
    for i in range(draw(st.integers(min_value=2, max_value=5))):
        ext = draw(st.sampled_from(list(extensions)))
        filename = f"normal_file_{i}{ext}"
        files.append((filename, False))

    # Add some files in ignored directories
    if ignored_dirs:
        for dirname in ignored_dirs:
            ext = draw(st.sampled_from(list(extensions)))
            filename = f"file_in_ignored{ext}"
            files.append((f"{dirname}/{filename}", True))

    # Add some files with ignored extensions (but in valid extensions set for scanner)
    # These won't be scanned anyway since they're not in the extensions set

    # Add some files matching ignored filenames
    for filename in ignored_files:
        if any(filename.endswith(ext) for ext in extensions):
            files.append((filename, True))

    # Add files in subdirectories that should not be ignored
    for i in range(draw(st.integers(min_value=1, max_value=3))):
        subdir = f"valid_dir_{i}"
        ext = draw(st.sampled_from(list(extensions)))
        filename = f"subfile{ext}"
        files.append((f"{subdir}/{filename}", False))

    return files, patterns, extensions


@given(data=ignore_pattern_strategy())
@settings(max_examples=100)
def test_ignore_pattern_exclusion(data):
    """
    **Feature: codebase-semantic-search, Property 3: Ignore Pattern Exclusion**
    **Validates: Requirements 1.4**

    For any file path and any ignore pattern set, if a path matches any pattern,
    FileScanner should exclude that file from results.
    """
    files, patterns, extensions = data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create the test directory structure
        for rel_path, _ in files:
            file_path = tmpdir_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f"# Content of {rel_path}\n", encoding="utf-8")

        # Create scanner with ignore patterns
        scanner = FileScanner(
            extensions=extensions,
            ignore_patterns=patterns,
        )

        # Scan the directory
        scanned_files = list(scanner.scan(tmpdir_path))
        scanned_paths = {sf.path for sf in scanned_files}

        # Property: Files that should be ignored must NOT appear in results
        for rel_path, should_be_ignored in files:
            file_path = (tmpdir_path / rel_path).resolve()
            ext = Path(rel_path).suffix.lower()

            # Only check files that have matching extensions
            if ext not in extensions:
                continue

            if should_be_ignored:
                assert file_path not in scanned_paths, (
                    f"File {rel_path} should be ignored by patterns {patterns} "
                    f"but was found in scan results"
                )
            else:
                assert file_path in scanned_paths, (
                    f"File {rel_path} should NOT be ignored but was not found in scan results"
                )


@st.composite
def gitignore_pattern_test_strategy(draw):
    """
    Generate test cases specifically for gitignore-style pattern matching.

    Tests various gitignore pattern syntaxes:
    - Simple directory names
    - Wildcard patterns (*.ext)
    - Double-star patterns (**/pattern)
    """
    extensions = {".py", ".js"}

    # Choose a pattern type to test
    pattern_type = draw(st.sampled_from(["simple_dir", "wildcard_ext", "nested_dir"]))

    if pattern_type == "simple_dir":
        # Test: directory name pattern excludes all files in that directory
        ignored_dir = "build"
        patterns = [ignored_dir]

        files = [
            ("src/main.py", False),
            ("build/output.py", True),
            ("build/sub/nested.py", True),
            ("other/build.py", False),  # File named build.py should NOT be ignored
        ]

    elif pattern_type == "wildcard_ext":
        # Test: *.ext pattern excludes files with that extension
        # Note: We need to use extensions that ARE in our filter set
        patterns = ["*.pyc", "__pycache__"]

        files = [
            ("main.py", False),
            ("util.js", False),
            ("__pycache__/cache.py", True),
        ]

    else:  # nested_dir
        # Test: nested directory patterns
        patterns = ["**/temp", "logs"]

        files = [
            ("src/main.py", False),
            ("logs/app.py", True),
            ("src/temp/cache.py", True),
            ("deep/nested/temp/file.py", True),
        ]

    return files, patterns, extensions


@given(data=gitignore_pattern_test_strategy())
@settings(max_examples=100)
def test_gitignore_pattern_matching(data):
    """
    **Feature: codebase-semantic-search, Property 3: Ignore Pattern Exclusion**
    **Validates: Requirements 1.4**

    Tests that FileScanner correctly handles gitignore-style patterns including
    simple directory names, wildcard patterns, and nested directory patterns.
    """
    files, patterns, extensions = data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create the test directory structure
        for rel_path, _ in files:
            file_path = tmpdir_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f"# Content of {rel_path}\n", encoding="utf-8")

        # Create scanner with ignore patterns
        scanner = FileScanner(
            extensions=extensions,
            ignore_patterns=patterns,
        )

        # Scan the directory
        scanned_files = list(scanner.scan(tmpdir_path))
        scanned_paths = {sf.path for sf in scanned_files}

        # Verify each file's expected presence/absence
        for rel_path, should_be_ignored in files:
            file_path = (tmpdir_path / rel_path).resolve()
            ext = Path(rel_path).suffix.lower()

            # Only check files that have matching extensions
            if ext not in extensions:
                continue

            if should_be_ignored:
                assert file_path not in scanned_paths, (
                    f"File {rel_path} should be ignored by patterns {patterns} "
                    f"but was found in scan results"
                )
            else:
                assert file_path in scanned_paths, (
                    f"File {rel_path} should NOT be ignored by patterns {patterns} "
                    f"but was not found in scan results"
                )
