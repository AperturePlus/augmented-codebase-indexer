"""
Property-based tests for GitignoreManager pattern parsing and matching.

**Feature: gitignore-path-traversal-fix**
"""

import tempfile
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.core.gitignore_manager import GitignoreManager, GitignorePattern


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Valid gitignore pattern characters (excluding special chars that need escaping)
pattern_chars = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
)

# Simple pattern name (like a filename or directory name)
simple_pattern_name = st.text(pattern_chars, min_size=1, max_size=20).filter(
    lambda s: not s.startswith(".") or len(s) > 1  # Avoid just "."
)

# Comment line (starts with #)
# Exclude control characters and surrogates (which can't be encoded in UTF-8)
comment_line = st.builds(
    lambda prefix, content: f"#{prefix}{content}",
    prefix=st.sampled_from(["", " ", "\t", "  "]),
    content=st.text(st.characters(blacklist_categories=["Cc", "Cs"]), min_size=0, max_size=50),
)

# Empty or whitespace-only line
empty_line = st.sampled_from(["", " ", "\t", "  ", "\t\t", "   "])

# Valid pattern line (non-empty, non-comment)
valid_pattern = simple_pattern_name


@st.composite
def gitignore_content_strategy(draw):
    """
    Generate gitignore file content with a mix of:
    - Valid patterns
    - Comment lines
    - Empty lines

    Returns:
        tuple: (content_string, expected_pattern_count)
    """
    lines = []
    expected_count = 0

    num_lines = draw(st.integers(min_value=0, max_value=20))

    for _ in range(num_lines):
        line_type = draw(st.sampled_from(["pattern", "comment", "empty"]))

        if line_type == "pattern":
            pattern = draw(valid_pattern)
            lines.append(pattern)
            expected_count += 1
        elif line_type == "comment":
            comment = draw(comment_line)
            lines.append(comment)
        else:  # empty
            empty = draw(empty_line)
            lines.append(empty)

    content = "\n".join(lines)
    return content, expected_count


# =============================================================================
# Property Tests
# =============================================================================


@given(data=gitignore_content_strategy())
@settings(max_examples=100, deadline=None)
def test_gitignore_comment_and_empty_line_filtering(data):
    """
    **Feature: gitignore-path-traversal-fix, Property 1: Gitignore Comment and Empty Line Filtering**
    **Validates: Requirements 1.1**

    For any .gitignore file content, the GitignoreManager SHALL only load
    non-comment, non-empty lines as patterns. Lines starting with # and
    blank lines SHALL be excluded.
    """
    content, expected_count = data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        gitignore_path = tmpdir_path / ".gitignore"

        # Write the generated content
        gitignore_path.write_text(content, encoding="utf-8")

        # Create manager and load patterns
        manager = GitignoreManager(tmpdir_path)
        loaded_count = manager.load_gitignore(gitignore_path)

        # Property: Number of loaded patterns equals expected count
        assert loaded_count == expected_count, (
            f"Expected {expected_count} patterns but loaded {loaded_count}. "
            f"Content:\n{content}"
        )

        # Property: All loaded patterns are non-empty and non-comments
        for pattern in manager.patterns:
            assert pattern.raw.strip(), "Loaded pattern should not be empty"
            assert not pattern.raw.startswith("#"), (
                f"Loaded pattern '{pattern.raw}' should not be a comment"
            )



@st.composite
def anchored_pattern_test_strategy(draw):
    """
    Generate test cases for root-anchored patterns (starting with /).

    Returns:
        tuple: (pattern_name, root_path_file, subdir_path_file, should_match_root, should_match_subdir)
    """
    # Generate a simple pattern name
    pattern_name = draw(simple_pattern_name)

    # Root-anchored pattern should only match at root level
    # /pattern matches root/pattern but NOT root/subdir/pattern
    return pattern_name


@given(pattern_name=anchored_pattern_test_strategy())
@settings(max_examples=100, deadline=None)
def test_root_anchored_pattern_matching(pattern_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 2: Root-Anchored Pattern Matching**
    **Validates: Requirements 1.2**

    For any gitignore pattern starting with `/`, the pattern SHALL only match
    paths at the root directory level, not in subdirectories.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .gitignore with anchored pattern
        gitignore_path = tmpdir_path / ".gitignore"
        gitignore_path.write_text(f"/{pattern_name}\n", encoding="utf-8")

        # Create manager and load patterns
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(gitignore_path)

        # Test 1: File at root level SHOULD match
        root_file = Path(pattern_name)
        assert manager.matches(root_file, is_dir=False), (
            f"Anchored pattern '/{pattern_name}' should match root-level file '{pattern_name}'"
        )

        # Test 2: Same filename in subdirectory should NOT match
        subdir_file = Path("subdir") / pattern_name
        assert not manager.matches(subdir_file, is_dir=False), (
            f"Anchored pattern '/{pattern_name}' should NOT match subdirectory file 'subdir/{pattern_name}'"
        )

        # Test 3: Deeply nested file should NOT match
        deep_file = Path("a") / "b" / "c" / pattern_name
        assert not manager.matches(deep_file, is_dir=False), (
            f"Anchored pattern '/{pattern_name}' should NOT match deeply nested file"
        )



@given(pattern_name=simple_pattern_name)
@settings(max_examples=100, deadline=None)
def test_directory_only_pattern_matching(pattern_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 3: Directory-Only Pattern Matching**
    **Validates: Requirements 1.3**

    For any gitignore pattern ending with `/`, the pattern SHALL match only
    directories with that name, not files with the same name.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .gitignore with directory-only pattern
        gitignore_path = tmpdir_path / ".gitignore"
        gitignore_path.write_text(f"{pattern_name}/\n", encoding="utf-8")

        # Create manager and load patterns
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(gitignore_path)

        # Test 1: Directory with that name SHOULD match
        dir_path = Path(pattern_name)
        assert manager.matches(dir_path, is_dir=True), (
            f"Directory-only pattern '{pattern_name}/' should match directory '{pattern_name}'"
        )

        # Test 2: File with the same name should NOT match
        file_path = Path(pattern_name)
        assert not manager.matches(file_path, is_dir=False), (
            f"Directory-only pattern '{pattern_name}/' should NOT match file '{pattern_name}'"
        )

        # Test 3: Directory in subdirectory SHOULD match (unanchored)
        subdir_dir = Path("subdir") / pattern_name
        assert manager.matches(subdir_dir, is_dir=True), (
            f"Directory-only pattern '{pattern_name}/' should match subdirectory 'subdir/{pattern_name}'"
        )

        # Test 4: File in subdirectory with same name should NOT match
        subdir_file = Path("subdir") / pattern_name
        assert not manager.matches(subdir_file, is_dir=False), (
            f"Directory-only pattern '{pattern_name}/' should NOT match file 'subdir/{pattern_name}'"
        )



@st.composite
def double_star_test_strategy(draw):
    """
    Generate test cases for double-star (**) glob patterns.

    Returns:
        tuple: (pattern, test_paths_with_expected_match)
    """
    # Generate a filename to match
    filename = draw(simple_pattern_name)
    assume(not filename.startswith("."))  # Avoid hidden files

    # Generate different depths of nesting
    depths = draw(st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=5))

    return filename, depths


@given(data=double_star_test_strategy())
@settings(max_examples=100, deadline=None)
def test_double_star_glob_matching(data):
    """
    **Feature: gitignore-path-traversal-fix, Property 4: Double-Star Glob Matching**
    **Validates: Requirements 1.4**

    For any gitignore pattern containing `**`, the pattern SHALL match zero
    or more directory levels at that position.
    """
    filename, depths = data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .gitignore with double-star pattern
        # Pattern: **/{filename} should match {filename} at any depth
        gitignore_path = tmpdir_path / ".gitignore"
        gitignore_path.write_text(f"**/{filename}\n", encoding="utf-8")

        # Create manager and load patterns
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(gitignore_path)

        # Test: File at various depths should all match
        for depth in depths:
            if depth == 0:
                # Root level
                test_path = Path(filename)
            else:
                # Nested at various depths
                parts = [f"dir{i}" for i in range(depth)]
                test_path = Path(*parts) / filename

            assert manager.matches(test_path, is_dir=False), (
                f"Pattern '**/{filename}' should match '{test_path}' at depth {depth}"
            )


@given(pattern_name=simple_pattern_name)
@settings(max_examples=100, deadline=None)
def test_double_star_prefix_pattern(pattern_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 4: Double-Star Glob Matching**
    **Validates: Requirements 1.4**

    Tests that patterns like '**/dirname' match directories at any depth.
    """
    assume(not pattern_name.startswith("."))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .gitignore with double-star prefix pattern
        gitignore_path = tmpdir_path / ".gitignore"
        gitignore_path.write_text(f"**/{pattern_name}\n", encoding="utf-8")

        # Create manager and load patterns
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(gitignore_path)

        # Test at root level
        assert manager.matches(Path(pattern_name), is_dir=False), (
            f"Pattern '**/{pattern_name}' should match root-level '{pattern_name}'"
        )

        # Test at depth 1
        assert manager.matches(Path("a") / pattern_name, is_dir=False), (
            f"Pattern '**/{pattern_name}' should match 'a/{pattern_name}'"
        )

        # Test at depth 3
        assert manager.matches(Path("a") / "b" / "c" / pattern_name, is_dir=False), (
            f"Pattern '**/{pattern_name}' should match 'a/b/c/{pattern_name}'"
        )



@st.composite
def negation_pattern_strategy(draw):
    """
    Generate test cases for negation patterns (!).

    Returns:
        tuple: (excluded_pattern, included_file, other_excluded_file)
    """
    # Generate a base pattern to exclude
    base_pattern = draw(simple_pattern_name)
    assume(len(base_pattern) >= 2)  # Need at least 2 chars for meaningful test

    # Generate a specific file to re-include
    included_file = draw(simple_pattern_name)
    assume(included_file != base_pattern)

    return base_pattern, included_file


@given(data=negation_pattern_strategy())
@settings(max_examples=100, deadline=None)
def test_negation_pattern_re_inclusion(data):
    """
    **Feature: gitignore-path-traversal-fix, Property 5: Negation Pattern Re-inclusion**
    **Validates: Requirements 1.5**

    For any gitignore pattern starting with `!`, the pattern SHALL re-include
    files that were previously excluded by earlier patterns.
    """
    base_pattern, included_file = data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .gitignore with:
        # 1. A pattern that excludes everything with a certain extension
        # 2. A negation pattern that re-includes a specific file
        gitignore_content = f"""*.txt
!{included_file}.txt
"""
        gitignore_path = tmpdir_path / ".gitignore"
        gitignore_path.write_text(gitignore_content, encoding="utf-8")

        # Create manager and load patterns
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(gitignore_path)

        # Test 1: Generic .txt file SHOULD be ignored
        generic_file = Path(f"{base_pattern}.txt")
        if base_pattern != included_file:
            assert manager.matches(generic_file, is_dir=False), (
                f"File '{generic_file}' should be ignored by '*.txt' pattern"
            )

        # Test 2: The specifically included file should NOT be ignored
        included_path = Path(f"{included_file}.txt")
        assert not manager.matches(included_path, is_dir=False), (
            f"File '{included_path}' should be re-included by '!{included_file}.txt' pattern"
        )


@given(pattern_name=simple_pattern_name)
@settings(max_examples=100, deadline=None)
def test_negation_pattern_order_matters(pattern_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 5: Negation Pattern Re-inclusion**
    **Validates: Requirements 1.5**

    Tests that the order of patterns matters - later patterns override earlier ones.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .gitignore where negation comes BEFORE exclusion
        # In this case, the file should still be excluded
        gitignore_content = f"""!{pattern_name}
{pattern_name}
"""
        gitignore_path = tmpdir_path / ".gitignore"
        gitignore_path.write_text(gitignore_content, encoding="utf-8")

        # Create manager and load patterns
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(gitignore_path)

        # The file should be ignored because the exclusion comes after negation
        test_path = Path(pattern_name)
        assert manager.matches(test_path, is_dir=False), (
            f"File '{pattern_name}' should be ignored because exclusion pattern comes after negation"
        )



@given(pattern_name=simple_pattern_name)
@settings(max_examples=100, deadline=None)
def test_relative_path_matching(pattern_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 6: Relative Path Matching**
    **Validates: Requirements 1.6**

    For any file path being matched, the GitignoreManager SHALL convert it to
    a path relative to the root directory before pattern matching.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create .gitignore with a simple pattern
        gitignore_path = tmpdir_path / ".gitignore"
        gitignore_path.write_text(f"{pattern_name}\n", encoding="utf-8")

        # Create manager and load patterns
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(gitignore_path)

        # Test 1: Relative path should match
        rel_path = Path(pattern_name)
        assert manager.matches(rel_path, is_dir=False), (
            f"Relative path '{rel_path}' should match pattern '{pattern_name}'"
        )

        # Test 2: Absolute path (under root) should also match
        abs_path = tmpdir_path / pattern_name
        assert manager.matches(abs_path, is_dir=False), (
            f"Absolute path '{abs_path}' should match pattern '{pattern_name}'"
        )

        # Test 3: Relative path in subdirectory should match (unanchored pattern)
        subdir_rel_path = Path("subdir") / pattern_name
        assert manager.matches(subdir_rel_path, is_dir=False), (
            f"Relative path in subdir '{subdir_rel_path}' should match pattern '{pattern_name}'"
        )

        # Test 4: Absolute path in subdirectory should match
        subdir_abs_path = tmpdir_path / "subdir" / pattern_name
        assert manager.matches(subdir_abs_path, is_dir=False), (
            f"Absolute path in subdir '{subdir_abs_path}' should match pattern '{pattern_name}'"
        )


@given(pattern_name=simple_pattern_name)
@settings(max_examples=100, deadline=None)
def test_path_outside_root_not_matched(pattern_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 6: Relative Path Matching**
    **Validates: Requirements 1.6**

    Tests that paths outside the root directory are not matched by patterns.
    """
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            tmpdir1_path = Path(tmpdir1).resolve()
            tmpdir2_path = Path(tmpdir2).resolve()

            # Create .gitignore in tmpdir1
            gitignore_path = tmpdir1_path / ".gitignore"
            gitignore_path.write_text(f"{pattern_name}\n", encoding="utf-8")

            # Create manager for tmpdir1
            manager = GitignoreManager(tmpdir1_path)
            manager.load_gitignore(gitignore_path)

            # Test: Path in tmpdir2 (outside root) should NOT be matched
            outside_path = tmpdir2_path / pattern_name
            assert not manager.matches(outside_path, is_dir=False), (
                f"Path outside root '{outside_path}' should NOT be matched"
            )


# =============================================================================
# Property Tests for Nested Gitignore Support
# =============================================================================

# Windows reserved device names that can't be used as directory names
WINDOWS_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})

# Directory name that's safe for all platforms (no trailing dots/spaces which Windows normalizes,
# and no Windows reserved device names)
safe_dir_name = simple_pattern_name.filter(
    lambda s: (
        not s.endswith(".") 
        and not s.endswith(" ")
        and s.upper() not in WINDOWS_RESERVED_NAMES
    )
)


@st.composite
def nested_gitignore_test_strategy(draw):
    """
    Generate test cases for nested gitignore pattern scoping.

    Returns:
        tuple: (subdir_name, pattern_name)
    """
    subdir_name = draw(safe_dir_name)
    pattern_name = draw(simple_pattern_name)
    assume(subdir_name != pattern_name)  # Avoid confusion
    return subdir_name, pattern_name


@given(data=nested_gitignore_test_strategy())
@settings(max_examples=100, deadline=None)
def test_nested_gitignore_pattern_scoping(data):
    """
    **Feature: gitignore-path-traversal-fix, Property 7: Nested Gitignore Pattern Scoping**
    **Validates: Requirements 2.1, 2.2**

    For any nested .gitignore file, patterns in that file SHALL be scoped
    relative to the directory containing the .gitignore, not the root directory.
    """
    subdir_name, pattern_name = data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create subdirectory structure
        subdir_path = tmpdir_path / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)

        # Create nested .gitignore in subdirectory
        nested_gitignore = subdir_path / ".gitignore"
        nested_gitignore.write_text(f"{pattern_name}\n", encoding="utf-8")

        # Create manager and load the nested gitignore
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(nested_gitignore)

        # Test 1: File in subdirectory SHOULD be matched by nested pattern
        subdir_file = Path(subdir_name) / pattern_name
        assert manager.matches(subdir_file, is_dir=False), (
            f"Pattern '{pattern_name}' in {subdir_name}/.gitignore should match "
            f"'{subdir_file}'"
        )

        # Test 2: Same filename at ROOT should NOT be matched by nested pattern
        root_file = Path(pattern_name)
        assert not manager.matches(root_file, is_dir=False), (
            f"Pattern '{pattern_name}' in {subdir_name}/.gitignore should NOT match "
            f"root-level file '{root_file}'"
        )

        # Test 3: Same filename in DIFFERENT subdirectory should NOT be matched
        other_subdir_file = Path("other_subdir") / pattern_name
        assert not manager.matches(other_subdir_file, is_dir=False), (
            f"Pattern '{pattern_name}' in {subdir_name}/.gitignore should NOT match "
            f"file in different subdirectory '{other_subdir_file}'"
        )


@given(subdir_name=safe_dir_name, pattern_name=simple_pattern_name)
@settings(max_examples=100, deadline=None)
def test_nested_gitignore_anchored_pattern_scoping(subdir_name, pattern_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 7: Nested Gitignore Pattern Scoping**
    **Validates: Requirements 2.1, 2.2**

    Tests that anchored patterns in nested .gitignore files are scoped
    relative to the nested directory, not the root.
    """
    assume(subdir_name != pattern_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create subdirectory structure with nested subdir
        subdir_path = tmpdir_path / subdir_name
        nested_subdir = subdir_path / "nested"
        nested_subdir.mkdir(parents=True, exist_ok=True)

        # Create nested .gitignore with anchored pattern
        nested_gitignore = subdir_path / ".gitignore"
        nested_gitignore.write_text(f"/{pattern_name}\n", encoding="utf-8")

        # Create manager and load the nested gitignore
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(nested_gitignore)

        # Test 1: File directly in subdir SHOULD match anchored pattern
        direct_file = Path(subdir_name) / pattern_name
        assert manager.matches(direct_file, is_dir=False), (
            f"Anchored pattern '/{pattern_name}' in {subdir_name}/.gitignore should match "
            f"'{direct_file}'"
        )

        # Test 2: File in nested subdir should NOT match anchored pattern
        nested_file = Path(subdir_name) / "nested" / pattern_name
        assert not manager.matches(nested_file, is_dir=False), (
            f"Anchored pattern '/{pattern_name}' in {subdir_name}/.gitignore should NOT match "
            f"'{nested_file}' (anchored patterns only match at scope root)"
        )


@st.composite
def gitignore_precedence_strategy(draw):
    """
    Generate test cases for gitignore pattern precedence.

    Returns:
        tuple: (subdir_name, pattern_name, root_excludes, nested_excludes)
    """
    subdir_name = draw(safe_dir_name)
    pattern_name = draw(simple_pattern_name)
    assume(subdir_name != pattern_name)
    
    # Decide whether root and nested gitignore exclude or include the pattern
    root_excludes = draw(st.booleans())
    nested_excludes = draw(st.booleans())
    
    return subdir_name, pattern_name, root_excludes, nested_excludes


@given(data=gitignore_precedence_strategy())
@settings(max_examples=100, deadline=None)
def test_gitignore_pattern_precedence(data):
    """
    **Feature: gitignore-path-traversal-fix, Property 8: Gitignore Pattern Precedence**
    **Validates: Requirements 2.3**

    For any path matched by multiple .gitignore files, patterns SHALL be applied
    in order from root to deepest directory, with later patterns taking precedence.
    """
    subdir_name, pattern_name, root_excludes, nested_excludes = data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create subdirectory
        subdir_path = tmpdir_path / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)

        # Create root .gitignore
        root_gitignore = tmpdir_path / ".gitignore"
        if root_excludes:
            root_gitignore.write_text(f"{pattern_name}\n", encoding="utf-8")
        else:
            root_gitignore.write_text(f"!{pattern_name}\n", encoding="utf-8")

        # Create nested .gitignore that may override
        nested_gitignore = subdir_path / ".gitignore"
        if nested_excludes:
            nested_gitignore.write_text(f"{pattern_name}\n", encoding="utf-8")
        else:
            nested_gitignore.write_text(f"!{pattern_name}\n", encoding="utf-8")

        # Create manager and load gitignores in order (root first, then nested)
        manager = GitignoreManager(tmpdir_path)
        manager.load_gitignore(root_gitignore)
        manager.load_gitignore(nested_gitignore)

        # Test: File in subdirectory should follow nested gitignore's decision
        subdir_file = Path(subdir_name) / pattern_name
        is_ignored = manager.matches(subdir_file, is_dir=False)

        # The nested gitignore should take precedence for files in its scope
        assert is_ignored == nested_excludes, (
            f"File '{subdir_file}' should be {'ignored' if nested_excludes else 'included'} "
            f"based on nested .gitignore (root_excludes={root_excludes}, nested_excludes={nested_excludes})"
        )


@given(pattern_name=simple_pattern_name)
@settings(max_examples=100, deadline=None)
def test_gitignore_hierarchy_loading(pattern_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 7: Nested Gitignore Pattern Scoping**
    **Validates: Requirements 2.1, 2.2**

    Tests that load_gitignore_hierarchy correctly loads all .gitignore files
    in the directory tree.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create directory structure with multiple .gitignore files
        subdir1 = tmpdir_path / "subdir1"
        subdir2 = tmpdir_path / "subdir2"
        nested = subdir1 / "nested"
        
        subdir1.mkdir()
        subdir2.mkdir()
        nested.mkdir()

        # Create .gitignore files at different levels
        (tmpdir_path / ".gitignore").write_text(f"root_{pattern_name}\n", encoding="utf-8")
        (subdir1 / ".gitignore").write_text(f"sub1_{pattern_name}\n", encoding="utf-8")
        (subdir2 / ".gitignore").write_text(f"sub2_{pattern_name}\n", encoding="utf-8")
        (nested / ".gitignore").write_text(f"nested_{pattern_name}\n", encoding="utf-8")

        # Create manager and load hierarchy
        manager = GitignoreManager(tmpdir_path)
        total_loaded = manager.load_gitignore_hierarchy()

        # Should have loaded 4 patterns (one from each .gitignore)
        assert total_loaded == 4, (
            f"Expected 4 patterns from hierarchy, got {total_loaded}"
        )

        # Verify patterns are properly scoped
        # Root pattern should match at root
        assert manager.matches(Path(f"root_{pattern_name}"), is_dir=False)
        
        # Subdir1 pattern should only match in subdir1
        assert manager.matches(Path("subdir1") / f"sub1_{pattern_name}", is_dir=False)
        assert not manager.matches(Path(f"sub1_{pattern_name}"), is_dir=False)
        
        # Subdir2 pattern should only match in subdir2
        assert manager.matches(Path("subdir2") / f"sub2_{pattern_name}", is_dir=False)
        assert not manager.matches(Path(f"sub2_{pattern_name}"), is_dir=False)
        
        # Nested pattern should only match in nested
        assert manager.matches(Path("subdir1") / "nested" / f"nested_{pattern_name}", is_dir=False)
        assert not manager.matches(Path("subdir1") / f"nested_{pattern_name}", is_dir=False)
