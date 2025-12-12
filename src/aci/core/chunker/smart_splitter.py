"""
Smart chunk splitter for oversized AST nodes.

Provides intelligent code splitting that avoids breaking syntax structures.
"""

import logging
import re
from typing import List

from aci.core.ast_parser import ASTNode
from aci.core.tokenizer import TokenizerInterface

from .models import CodeChunk

logger = logging.getLogger(__name__)


class SmartChunkSplitter:
    """
    智能代码拆分器 - 避免破坏语法结构

    拆分策略（按优先级）:
    1. 在空行处拆分
    2. 在语句边界拆分（基于缩进和语法模式）
    3. 在完整行处拆分（保证不切断语句中间）
    4. 最后手段：硬截断（记录警告）

    每个子块会添加上下文前缀（如类名、函数签名）以保持语义完整性
    """

    # Patterns that indicate statement boundaries
    STATEMENT_BOUNDARY_PATTERNS = [
        re.compile(r"^\s*def\s+"),  # Function definition
        re.compile(r"^\s*class\s+"),  # Class definition
        re.compile(r"^\s*if\s+"),  # If statement
        re.compile(r"^\s*elif\s+"),  # Elif statement
        re.compile(r"^\s*else\s*:"),  # Else statement
        re.compile(r"^\s*for\s+"),  # For loop
        re.compile(r"^\s*while\s+"),  # While loop
        re.compile(r"^\s*try\s*:"),  # Try block
        re.compile(r"^\s*except\s*"),  # Except block
        re.compile(r"^\s*finally\s*:"),  # Finally block
        re.compile(r"^\s*with\s+"),  # With statement
        re.compile(r"^\s*return\s"),  # Return statement
        re.compile(r"^\s*yield\s"),  # Yield statement
        re.compile(r"^\s*raise\s"),  # Raise statement
        re.compile(r"^\s*@"),  # Decorator
    ]

    def __init__(self, tokenizer: TokenizerInterface):
        """
        Initialize the SmartChunkSplitter.

        Args:
            tokenizer: Tokenizer for token counting
        """
        self._tokenizer = tokenizer

    def split_oversized_node(
        self,
        node: ASTNode,
        max_tokens: int,
        file_path: str,
        language: str,
        base_metadata: dict,
        docstring_prefix: str = "",
    ) -> List[CodeChunk]:
        """
        拆分超大 AST 节点

        Args:
            node: The oversized AST node to split
            max_tokens: Maximum tokens per chunk
            file_path: Path to the source file
            language: Programming language
            base_metadata: Base metadata for chunks
            docstring_prefix: Docstring prefix to add to first chunk

        Returns:
            List of CodeChunk objects with correct line numbers
        """
        lines = node.content.split("\n")

        # Generate context prefix for semantic completeness
        context_prefix = self._generate_context_prefix(node)
        context_prefix_tokens = (
            self._tokenizer.count_tokens(context_prefix) if context_prefix else 0
        )

        # Effective max tokens for content (accounting for prefix)
        effective_max_tokens = max_tokens - context_prefix_tokens
        if effective_max_tokens < 50:
            # If prefix is too large relative to max_tokens, skip it
            context_prefix = ""
            effective_max_tokens = max_tokens

        # Find split points using priority strategy
        split_points = self._find_split_points(lines, effective_max_tokens)

        # Create chunks from split points
        chunks = self._create_chunks_from_splits(
            lines=lines,
            split_points=split_points,
            node=node,
            file_path=file_path,
            language=language,
            base_metadata=base_metadata,
            context_prefix=context_prefix,
            max_tokens=max_tokens,
            docstring_prefix=docstring_prefix,
        )

        return chunks

    def _generate_context_prefix(self, node: ASTNode) -> str:
        """
        Generate a context prefix for split chunks.

        The prefix helps maintain semantic context when a large node
        is split into multiple chunks.

        Args:
            node: The AST node being split

        Returns:
            Context prefix string (may be empty)
        """
        if node.node_type == "method" and node.parent_name:
            return f"# Context: class {node.parent_name}\n"
        elif node.node_type == "function":
            return f"# Context: function {node.name}\n"
        elif node.node_type == "class":
            return f"# Context: class {node.name}\n"
        return ""

    def _find_split_points(
        self,
        lines: List[str],
        max_tokens: int,
    ) -> List[int]:
        """
        Find optimal split points in the code.

        Strategy (by priority):
        1. Empty lines
        2. Statement boundaries (based on indentation/patterns)
        3. Any complete line

        Args:
            lines: List of code lines
            max_tokens: Maximum tokens per chunk

        Returns:
            List of line indices where splits should occur
        """
        split_points = [0]  # Always start at the beginning
        current_start = 0

        while current_start < len(lines):
            # Find the farthest point we can go without exceeding token limit
            end_idx = self._find_max_end_index(lines, current_start, max_tokens)

            if end_idx <= current_start:
                # Single line exceeds limit - force split after this line
                logger.warning(f"Line {current_start + 1} exceeds token limit, forcing split")
                end_idx = current_start + 1

            if end_idx >= len(lines):
                break

            # Find the best split point within the valid range
            best_split = self._find_best_split_point(lines, current_start, end_idx)

            if best_split > current_start:
                split_points.append(best_split)
                current_start = best_split
            else:
                split_points.append(end_idx)
                current_start = end_idx

        return split_points

    def _find_max_end_index(
        self,
        lines: List[str],
        start_idx: int,
        max_tokens: int,
    ) -> int:
        """
        Find the maximum end index that fits within token limit.

        Uses binary search for efficiency.
        """
        left = start_idx
        right = len(lines)

        while left < right:
            mid = (left + right + 1) // 2
            content = "\n".join(lines[start_idx:mid])
            tokens = self._tokenizer.count_tokens(content)

            if tokens <= max_tokens:
                left = mid
            else:
                right = mid - 1

        return left

    def _find_best_split_point(
        self,
        lines: List[str],
        start_idx: int,
        end_idx: int,
    ) -> int:
        """
        Find the best split point within a range.

        Priority:
        1. Empty lines (highest priority)
        2. Statement boundaries
        3. Lines with lower indentation (block boundaries)
        """
        empty_line_candidates = []
        statement_boundary_candidates = []
        low_indent_candidates = []

        base_indent = self._get_indentation(lines[start_idx]) if start_idx < len(lines) else 0

        for i in range(end_idx - 1, start_idx, -1):
            if i >= len(lines):
                continue

            line = lines[i]

            if not line.strip():
                empty_line_candidates.append(i + 1)
                continue

            if self._is_statement_boundary(line):
                statement_boundary_candidates.append(i)

            current_indent = self._get_indentation(line)
            if current_indent <= base_indent and line.strip():
                low_indent_candidates.append(i)

        if empty_line_candidates:
            return empty_line_candidates[0]
        if statement_boundary_candidates:
            return statement_boundary_candidates[0]
        if low_indent_candidates:
            return low_indent_candidates[0]

        return end_idx

    def _is_statement_boundary(self, line: str) -> bool:
        """Check if a line represents a statement boundary."""
        for pattern in self.STATEMENT_BOUNDARY_PATTERNS:
            if pattern.match(line):
                return True
        return False

    def _get_indentation(self, line: str) -> int:
        """Get the indentation level of a line."""
        return len(line) - len(line.lstrip())

    def _create_chunks_from_splits(
        self,
        lines: List[str],
        split_points: List[int],
        node: ASTNode,
        file_path: str,
        language: str,
        base_metadata: dict,
        context_prefix: str,
        max_tokens: int,
        docstring_prefix: str = "",
    ) -> List[CodeChunk]:
        """Create CodeChunk objects from split points."""
        chunks = []

        for i, start_idx in enumerate(split_points):
            if i + 1 < len(split_points):
                end_idx = split_points[i + 1]
            else:
                end_idx = len(lines)

            if start_idx >= end_idx:
                continue

            chunk_lines = lines[start_idx:end_idx]
            chunk_content = "\n".join(chunk_lines)

            # Add docstring to first chunk, context prefix to continuation chunks
            if i == 0 and docstring_prefix:
                chunk_content = f"{docstring_prefix}{chunk_content}"
            elif i > 0 and context_prefix:
                chunk_content = context_prefix + chunk_content

            token_count = self._tokenizer.count_tokens(chunk_content)
            if token_count > max_tokens:
                logger.warning(
                    f"Chunk exceeds token limit ({token_count} > {max_tokens}), "
                    f"this may indicate a very long single line"
                )

            metadata = base_metadata.copy()
            metadata["function_name"] = node.name
            if node.parent_name:
                metadata["parent_class"] = node.parent_name
            metadata["is_partial"] = True
            metadata["part_index"] = i
            metadata["total_parts"] = len(split_points)
            if docstring_prefix:
                metadata["docstring_included_in_chunk"] = i == 0
            if i > 0:
                metadata["has_context_prefix"] = bool(context_prefix)

            chunk_start_line = node.start_line + start_idx
            chunk_end_line = node.start_line + end_idx - 1

            chunk = CodeChunk(
                file_path=file_path,
                start_line=chunk_start_line,
                end_line=chunk_end_line,
                content=chunk_content,
                language=language,
                chunk_type=node.node_type,
                metadata=metadata,
            )
            chunks.append(chunk)

        # If only one chunk, remove partial markers
        if len(chunks) == 1:
            chunks[0].metadata.pop("is_partial", None)
            chunks[0].metadata.pop("part_index", None)
            chunks[0].metadata.pop("total_parts", None)
            chunks[0].metadata.pop("docstring_included_in_chunk", None)

        return chunks
