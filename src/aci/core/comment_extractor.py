"""
Comment Extractor - Extracts documentation comments from source code.

Uses strategy pattern for language-specific extraction and heuristic
algorithms for robust comment-to-declaration association.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CommentCandidate:
    """A candidate comment that might be a doc comment."""
    text: str
    end_line: int  # 0-based line number where comment ends
    end_byte: int  # Byte position where comment ends
    score: float = 0.0  # Heuristic score (higher = more likely to be doc comment)


class CommentExtractorStrategy(ABC):
    """Abstract base class for language-specific comment extraction."""

    @abstractmethod
    def extract(
        self, node: Any, content: str, root_node: Any, node_name: str | None = None
    ) -> str | None:
        """Extract doc comment for a node."""
        pass

    def _get_preceding_content(self, node: Any, content: str) -> str:
        """Get content before the node."""
        return content[:node.start_byte]

    def _get_node_start_line(self, node: Any) -> int:
        """Get 0-based start line of node."""
        return node.start_point[0]


class HeuristicScorer:
    """
    Scores comment candidates using heuristics.

    Heuristics:
    1. Line distance: Closer comments score higher, too far = reject
    2. No blank lines: Comments with blank lines between them and declaration are rejected
    3. Content match: Comments mentioning the declaration name score higher
    4. Doc comment style: /** or /// style comments score higher
    """

    MAX_LINE_DISTANCE = 2  # Max lines between comment end and declaration start

    def score(
        self,
        candidate: CommentCandidate,
        node_start_line: int,
        node_start_byte: int,
        node_name: str | None = None,
        has_blank_line_between: bool = False,
    ) -> float:
        """
        Calculate heuristic score for a comment candidate.

        Returns score between 0.0 and 1.0.
        A score below 0.5 means the comment should be rejected.
        """
        # Line distance check - reject if too far
        line_distance = node_start_line - candidate.end_line
        if line_distance <= 0:
            return 0.0  # Comment after declaration
        if line_distance > self.MAX_LINE_DISTANCE:
            return 0.0  # Comment too far from declaration

        # Blank line between comment and declaration - reject
        if has_blank_line_between:
            return 0.0

        # Base score for being close enough
        score = 0.5

        # Closer comments score higher (0.0 - 0.2)
        score += 0.2 * (1.0 - (line_distance - 1) / self.MAX_LINE_DISTANCE)

        # Content match heuristic (0.0 - 0.2)
        if node_name and isinstance(node_name, str) and node_name.lower() in candidate.text.lower():
            score += 0.2

        # Doc comment style bonus (0.0 - 0.1)
        text_stripped = candidate.text.strip()
        if text_stripped.startswith('/**') or text_stripped.startswith('///'):
            score += 0.1

        return min(score, 1.0)


class JSDocExtractor(CommentExtractorStrategy):
    """Extracts JSDoc comments (/** ... */) for JavaScript/TypeScript/Java."""

    JSDOC_PATTERN = re.compile(r'/\*\*[\s\S]*?\*/', re.MULTILINE)

    # Keywords that can appear between JSDoc and declaration
    ALLOWED_KEYWORDS = {
        'export', 'default', 'async', 'static', 'public', 'private', 'protected',
        'abstract', 'readonly', 'override', 'declare', 'const', 'let', 'var',
        'function', 'class', 'interface', 'type', 'enum', 'namespace',
    }

    def __init__(self):
        self.scorer = HeuristicScorer()

    def extract(
        self, node: Any, content: str, root_node: Any, node_name: str | None = None
    ) -> str | None:
        """Extract JSDoc comment preceding a node."""
        preceding = self._get_preceding_content(node, content)
        node_start_line = self._get_node_start_line(node)
        node_start_byte = node.start_byte

        # Find all JSDoc comments before this node
        matches = list(self.JSDOC_PATTERN.finditer(preceding))
        if not matches:
            return None

        # Get the last (closest) match
        last_match = matches[-1]
        comment_text = last_match.group()
        comment_end_byte = last_match.end()

        # Check what's between comment and node
        between = preceding[comment_end_byte:]
        has_blank = '\n\n' in between or '\n\r\n' in between

        # Check if content between comment and node is acceptable
        # Allow: whitespace, newlines, and certain keywords (export, async, etc.)
        between_stripped = between.strip()
        if between_stripped:
            # Check if all tokens between are allowed keywords
            tokens = between_stripped.split()
            if not all(token in self.ALLOWED_KEYWORDS for token in tokens):
                return None

        # Calculate end line of comment
        comment_end_line = preceding[:comment_end_byte].count('\n')

        # Score the candidate
        candidate = CommentCandidate(
            text=comment_text,
            end_line=comment_end_line,
            end_byte=comment_end_byte,
        )

        # Extract node name for scoring (if available)
        if node_name is None:
            # Handle both real tree-sitter nodes and mock objects
            if hasattr(node, 'name'):
                name_attr = getattr(node, 'name', None)
                if name_attr is not None:
                    node_name = str(name_attr) if not isinstance(name_attr, str) else name_attr

        score = self.scorer.score(
            candidate,
            node_start_line,
            node_start_byte,
            node_name=node_name,
            has_blank_line_between=has_blank,
        )

        # Threshold for accepting (raised to prevent distant matches)
        if score < 0.5:
            return None

        return comment_text


class GoDocExtractor(CommentExtractorStrategy):
    """Extracts Go doc comments (// lines) for Go."""

    def __init__(self):
        self.scorer = HeuristicScorer()

    def _first_comment_line_matches(self, comment_lines: list[str], node_name: str | None) -> bool:
        """Check whether the first comment line starts with the identifier name."""
        if not node_name or not comment_lines:
            return True
        cleaned = comment_lines[0].lstrip('/').strip()
        # Strip pointer/receiver punctuation and compare case-insensitively
        cleaned = cleaned.lstrip('*(').rstrip(')')
        return cleaned.lower().startswith(node_name.lower())

    def extract(
        self, node: Any, content: str, root_node: Any, node_name: str | None = None
    ) -> str | None:
        """
        Extract Go doc comment preceding a node.

        In Go, doc comments must be immediately before the declaration with no
        blank lines in between. A blank line breaks the association.
        """
        preceding = self._get_preceding_content(node, content)
        lines = preceding.split('\n')
        comment_lines: list[str] = []

        start_idx = len(lines) - 1

        # Skip the partial line at the end (content before node on same line)
        if start_idx >= 0 and lines[start_idx].strip() == '':
            start_idx -= 1

        blank_line_count = 0
        while start_idx >= 0 and lines[start_idx].strip() == '':
            blank_line_count += 1
            start_idx -= 1

        if blank_line_count > 0:
            return None

        for i in range(start_idx, -1, -1):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith('//'):
                comment_lines.insert(0, stripped)
            elif stripped == '':
                break
            else:
                break

        if comment_lines:
            if node_name and not self._first_comment_line_matches(comment_lines, node_name):
                return None
            return '\n'.join(comment_lines)

        # Fallback: handle block comments immediately preceding the declaration
        block_match = re.search(r'/\*[\s\S]*?\*/\s*$', preceding, re.MULTILINE)
        if block_match:
            comment_text = block_match.group()
            tail = preceding[block_match.end():]
            if tail.strip():
                return None
            if node_name:
                cleaned = comment_text.strip().lstrip('/*').strip()
                if not cleaned.lower().startswith(node_name.lower()):
                    return None
            return comment_text

        return None


class DoxygenExtractor(CommentExtractorStrategy):
    """Extracts Doxygen comments (/** */ or ///) for C/C++."""

    BLOCK_PATTERN = re.compile(r'/\*\*[\s\S]*?\*/', re.MULTILINE)
    LINE_PATTERN = re.compile(r'^\s*///.*$', re.MULTILINE)

    def __init__(self):
        self.scorer = HeuristicScorer()

    def extract(
        self, node: Any, content: str, root_node: Any, node_name: str | None = None
    ) -> str | None:
        """Extract Doxygen comment preceding a node."""
        preceding = self._get_preceding_content(node, content)

        # Try line comments first (///)
        lines = preceding.split('\n')
        comment_lines = []

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith('///') or stripped.startswith('//!'):
                comment_lines.insert(0, stripped)
            elif stripped == '':
                if comment_lines:
                    break
            else:
                break

        if comment_lines:
            return '\n'.join(comment_lines)

        # Fall back to block comments (/** */)
        matches = list(self.BLOCK_PATTERN.finditer(preceding))
        if matches:
            last_match = matches[-1]
            between = preceding[last_match.end():]
            if not between.strip():
                return last_match.group()

        return None


class CommentExtractor:
    """
    Main comment extractor using strategy pattern.

    Delegates to language-specific extractors.
    """

    def __init__(self):
        self._extractors = {
            'javascript': JSDocExtractor(),
            'typescript': JSDocExtractor(),
            'java': JSDocExtractor(),
            'go': GoDocExtractor(),
            'c': DoxygenExtractor(),
            'cpp': DoxygenExtractor(),
        }

    def extract(
        self,
        node: Any,
        content: str,
        root_node: Any,
        language: str,
        node_name: str | None = None,
    ) -> str | None:
        """Extract doc comment for a node in the given language."""
        extractor = self._extractors.get(language)
        if extractor:
            return extractor.extract(node, content, root_node, node_name=node_name)
        return None


# Global instance
_extractor = CommentExtractor()


def extract_jsdoc(
    node: Any, content: str, root_node: Any, node_name: str | None = None
) -> str | None:
    """Extract JSDoc comment for a JavaScript/TypeScript node."""
    return _extractor.extract(node, content, root_node, 'javascript', node_name=node_name)


def extract_go_doc(
    node: Any, content: str, root_node: Any, node_name: str | None = None
) -> str | None:
    """Extract Go doc comment for a Go node."""
    return _extractor.extract(node, content, root_node, 'go', node_name=node_name)


def extract_javadoc(
    node: Any, content: str, root_node: Any, node_name: str | None = None
) -> str | None:
    """Extract Javadoc comment for a Java node."""
    return _extractor.extract(node, content, root_node, 'java', node_name=node_name)


def extract_doxygen(
    node: Any, content: str, root_node: Any, node_name: str | None = None
) -> str | None:
    """Extract Doxygen comment for a C/C++ node."""
    return _extractor.extract(node, content, root_node, 'cpp', node_name=node_name)
