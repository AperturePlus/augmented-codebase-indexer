"""
Property-based tests for Comment Extractor.

Uses Hypothesis to generate test cases and verify invariants.
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import Mock

from aci.core.comment_extractor import (
    HeuristicScorer,
    JSDocExtractor,
    GoDocExtractor,
    CommentCandidate,
)


class TestHeuristicScorerProperties:
    """Property-based tests for HeuristicScorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = HeuristicScorer()

    @given(
        comment_end_line=st.integers(min_value=0, max_value=100),
        node_start_line=st.integers(min_value=0, max_value=100),
        node_name=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        has_blank_line=st.booleans(),
    )
    def test_score_range_invariant(self, comment_end_line, node_start_line, node_name, has_blank_line):
        """Test that scores are always in valid range [0.0, 1.0]."""
        candidate = CommentCandidate(
            text=f"/** Test comment with {node_name or 'nothing'} */",
            end_line=comment_end_line,
            end_byte=comment_end_line * 50,  # Approximate
        )
        
        score = self.scorer.score(
            candidate,
            node_start_line,
            node_start_line * 50,
            node_name=node_name,
            has_blank_line_between=has_blank_line,
        )
        
        assert 0.0 <= score <= 1.0

    @given(
        comment_end_line=st.integers(min_value=1, max_value=50),
        node_name=st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    )
    def test_name_match_increases_score(self, comment_end_line, node_name):
        """Test that matching node name in comment increases score."""
        assume(len(node_name) >= 3)  # Ensure meaningful names
        
        # Comment with name mention
        comment_with_name = f"/** Function {node_name} does something */"
        candidate_with = CommentCandidate(
            text=comment_with_name,
            end_line=comment_end_line,
            end_byte=comment_end_line * 50,
        )
        
        # Comment without name mention
        comment_without_name = "/** Function does something */"
        candidate_without = CommentCandidate(
            text=comment_without_name,
            end_line=comment_end_line,
            end_byte=comment_end_line * 50,
        )
        
        node_start_line = comment_end_line + 1
        
        score_with = self.scorer.score(
            candidate_with, node_start_line, node_start_line * 50,
            node_name=node_name, has_blank_line_between=False
        )
        score_without = self.scorer.score(
            candidate_without, node_start_line, node_start_line * 50,
            node_name=node_name, has_blank_line_between=False
        )
        
        # If both scores are non-zero, name match should increase score
        if score_with > 0 and score_without > 0:
            assert score_with >= score_without

    @given(comment_end_line=st.integers(min_value=1, max_value=50))
    def test_blank_line_always_rejects(self, comment_end_line):
        """Test that blank lines always result in score 0.0."""
        candidate = CommentCandidate(
            text="/** Any comment */",
            end_line=comment_end_line,
            end_byte=comment_end_line * 50,
        )
        
        score = self.scorer.score(
            candidate,
            comment_end_line + 1,
            (comment_end_line + 1) * 50,
            node_name="test",
            has_blank_line_between=True,  # Blank line present
        )
        
        assert score == 0.0

    @given(
        comment_end_line=st.integers(min_value=1, max_value=50),
        distance=st.integers(min_value=3, max_value=20),  # Beyond MAX_LINE_DISTANCE
    )
    def test_distant_comments_rejected(self, comment_end_line, distance):
        """Test that comments too far from declaration are rejected."""
        candidate = CommentCandidate(
            text="/** Distant comment */",
            end_line=comment_end_line,
            end_byte=comment_end_line * 50,
        )
        
        node_start_line = comment_end_line + distance
        
        score = self.scorer.score(
            candidate,
            node_start_line,
            node_start_line * 50,
            node_name="test",
            has_blank_line_between=False,
        )
        
        # Should be rejected due to distance
        assert score == 0.0

    @given(
        comment_style=st.sampled_from(["/**", "///", "/*", "//"]),
        comment_content=st.text(min_size=1, max_size=100),
    )
    def test_doc_comment_style_bonus(self, comment_style, comment_content):
        """Test that doc comment styles get bonus points."""
        doc_comment = f"{comment_style} {comment_content} */" if comment_style.startswith("/*") else f"{comment_style} {comment_content}"
        regular_comment = f"/* {comment_content} */"
        
        doc_candidate = CommentCandidate(text=doc_comment, end_line=5, end_byte=100)
        regular_candidate = CommentCandidate(text=regular_comment, end_line=5, end_byte=100)
        
        doc_score = self.scorer.score(doc_candidate, 6, 120, "test", False)
        regular_score = self.scorer.score(regular_candidate, 6, 120, "test", False)
        
        # Doc styles (/** and ///) should score higher than regular (/* and //)
        if comment_style in ["/**", "///"]:
            if doc_score > 0 and regular_score > 0:
                assert doc_score >= regular_score


class TestJSDocExtractorProperties:
    """Property-based tests for JSDoc extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = JSDocExtractor()

    @given(
        comment_text=st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(blacklist_categories=["Cs", "Cc"])
        ),
        function_name=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        keywords=st.lists(
            st.sampled_from(['export', 'async', 'static', 'public', 'const']),
            min_size=0, max_size=3, unique=True
        ),
    )
    def test_jsdoc_extraction_with_keywords(self, comment_text, function_name, keywords):
        """Test JSDoc extraction with various keyword combinations."""
        assume(function_name.isidentifier())  # Valid identifier
        # Avoid comment text that would break JSDoc structure
        assume('*/' not in comment_text)
        assume('/*' not in comment_text)
        
        keyword_str = " ".join(keywords) + " " if keywords else ""
        content = f"""/** {comment_text} */
{keyword_str}function {function_name}() {{
    return 1;
}}"""
        
        node = Mock()
        node.start_byte = content.find(keyword_str + "function" if keyword_str else "function")
        node.start_point = (content[:node.start_byte].count('\n'), 0)
        
        result = self.extractor.extract(node, content, None)
        
        # Should extract comment when only allowed keywords are present
        if all(kw in self.extractor.ALLOWED_KEYWORDS for kw in keywords):
            assert result is not None
            assert comment_text in result
        # Note: We can't assert None for disallowed keywords since our sampled_from
        # only includes allowed keywords

    @given(
        comment_lines=st.lists(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(blacklist_categories=["Cs", "Cc"])
            ),
            min_size=1, max_size=10
        ),
        function_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    )
    def test_multiline_jsdoc_extraction(self, comment_lines, function_name):
        """Test extraction of multi-line JSDoc comments."""
        assume(function_name.isidentifier())
        # Avoid lines that could break JSDoc structure or look like code
        for line in comment_lines:
            assume('*/' not in line)
            assume('/*' not in line)
            # Avoid lines that look like function declarations
            assume(not line.strip().startswith('function'))
        
        # Build multi-line JSDoc
        jsdoc_lines = ["/**"]
        for line in comment_lines:
            jsdoc_lines.append(f" * {line}")
        jsdoc_lines.append(" */")
        
        content = "\n".join(jsdoc_lines) + f"\nfunction {function_name}() {{\n    return 1;\n}}"
        
        node = Mock()
        node.start_byte = content.find("function")
        node.start_point = (content[:node.start_byte].count('\n'), 0)
        
        result = self.extractor.extract(node, content, None)
        
        assert result is not None
        # Should contain the original comment structure
        for line in comment_lines:
            if line.strip():  # Skip empty lines
                assert line in result

    @given(
        valid_content=st.booleans(),
        has_blank_line=st.booleans(),
    )
    def test_jsdoc_rejection_conditions(self, valid_content, has_blank_line):
        """Test conditions under which JSDoc should be rejected."""
        if valid_content:
            between_content = "export async"  # Valid keywords
        else:
            between_content = "someVariable = 5;"  # Invalid content
        
        blank_line = "\n\n" if has_blank_line else "\n"
        
        content = f"""/** Test comment */
{between_content}{blank_line}function test() {{
    return 1;
}}"""
        
        node = Mock()
        node.start_byte = content.find("function")
        # Calculate line number based on content
        lines_before = content[:content.find("function")].count('\n')
        node.start_point = (lines_before, 0)
        
        result = self.extractor.extract(node, content, None)
        
        # Should be rejected if invalid content or blank line
        if not valid_content or has_blank_line:
            assert result is None
        else:
            assert result is not None


class TestGoDocExtractorProperties:
    """Property-based tests for Go doc extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = GoDocExtractor()

    @given(
        comment_lines=st.lists(
            st.text(
                min_size=1,
                max_size=80,
                alphabet=st.characters(blacklist_categories=["Cs", "Cc"])
            ),
            min_size=1, max_size=15
        ),
        function_name=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        has_blank_line=st.booleans(),
    )
    def test_go_doc_extraction_with_blank_lines(self, comment_lines, function_name, has_blank_line):
        """Test Go doc extraction behavior with blank lines."""
        assume(function_name.isidentifier())
        
        # Build Go doc comments
        doc_lines = [f"// {line}" for line in comment_lines]
        
        if has_blank_line:
            content_lines = doc_lines + ["", f"func {function_name}() {{", "}"]
        else:
            content_lines = doc_lines + [f"func {function_name}() {{", "}"]
        
        content = "\n".join(content_lines)
        
        node = Mock()
        node.start_byte = content.rfind("func")
        func_line = len(doc_lines) + (1 if has_blank_line else 0)
        node.start_point = (func_line, 0)
        
        result = self.extractor.extract(node, content, None)
        
        # Should be rejected if blank line exists
        if has_blank_line:
            assert result is None
        else:
            assert result is not None
            # Result contains lines with "// " prefix, and trailing whitespace may be stripped
            for line in comment_lines:
                stripped_line = line.strip()
                if stripped_line:
                    # The result format is "// line", check if the stripped line content appears
                    assert f"// {stripped_line}" in result or stripped_line in result

    @given(
        doc_lines=st.lists(
            st.text(
                min_size=2,
                max_size=50,
                alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            ),
            min_size=1,
            max_size=5,
        ),
        separator_lines=st.lists(
            st.text(
                min_size=2,
                max_size=50,
                alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            ),
            min_size=1,
            max_size=3,
        ),
        function_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    )
    def test_go_doc_stops_at_non_comment(self, doc_lines, separator_lines, function_name):
        """Test that Go doc extraction stops at non-comment lines."""
        assume(function_name.isidentifier())
        # Ensure separator lines are truly distinct from doc lines (not substrings)
        assume(set(doc_lines).isdisjoint(set(separator_lines)))
        # Ensure no line is a substring of another to avoid false positives
        for sep_line in separator_lines:
            for doc_line in doc_lines:
                assume(sep_line.strip() not in doc_line)
                assume(doc_line.strip() not in sep_line)
        
        # Build content with separate comment blocks
        first_block = [f"// {line}" for line in separator_lines]
        separator = ["package main"]  # Non-comment line
        second_block = [f"// {line}" for line in doc_lines]
        func_line = [f"func {function_name}() {{", "}"]
        
        all_lines = first_block + separator + second_block + func_line
        content = "\n".join(all_lines)
        
        node = Mock()
        node.start_byte = content.find("func")
        func_line_num = len(first_block) + len(separator) + len(second_block)
        node.start_point = (func_line_num, 0)
        
        result = self.extractor.extract(node, content, None)

        if result is not None:
            result_lines = result.split('\n')
            # Should only contain the second block (immediately before function)
            for line in doc_lines:
                stripped = line.strip()
                if stripped:
                    expected_line = f"// {stripped}"
                    assert any(expected_line == rl.strip() for rl in result_lines)

            # Should NOT contain the first block (separated by non-comment)
            for line in separator_lines:
                stripped = line.strip()
                if stripped:
                    excluded_line = f"// {stripped}"
                    assert all(excluded_line != rl.strip() for rl in result_lines)


class TestCommentExtractionInvariants:
    """Test invariants that should hold across all extractors."""

    @given(
        extractor_type=st.sampled_from(['jsdoc', 'godoc']),
        content_size=st.integers(min_value=10, max_value=1000),
        node_position=st.integers(min_value=5, max_value=500),
    )
    def test_extraction_never_crashes(self, extractor_type, content_size, node_position):
        """Test that extraction never crashes regardless of input."""
        # Generate some content
        content = "a" * content_size
        
        # Ensure node position is within content
        assume(node_position < len(content))
        
        node = Mock()
        node.start_byte = node_position
        node.start_point = (node_position // 50, node_position % 50)  # Approximate line/col
        
        if extractor_type == 'jsdoc':
            extractor = JSDocExtractor()
        else:
            extractor = GoDocExtractor()
        
        # Should not crash, regardless of result
        try:
            result = extractor.extract(node, content, None)
            # Result can be None or string, but should not crash
            assert result is None or isinstance(result, str)
        except Exception as e:
            pytest.fail(f"Extraction crashed with {type(e).__name__}: {e}")

    @given(
        comment_text=st.text(min_size=0, max_size=500),
        node_byte_pos=st.integers(min_value=0, max_value=1000),
    )
    def test_result_type_invariant(self, comment_text, node_byte_pos):
        """Test that extraction always returns None or non-empty string."""
        content = f"/** {comment_text} */\nfunction test() {{}}"
        
        # Ensure node position is reasonable
        assume(node_byte_pos < len(content))
        
        node = Mock()
        node.start_byte = node_byte_pos
        node.start_point = (1, 0)
        
        extractor = JSDocExtractor()
        result = extractor.extract(node, content, None)
        
        # Result must be None or non-empty string
        assert result is None or (isinstance(result, str) and len(result) > 0)
