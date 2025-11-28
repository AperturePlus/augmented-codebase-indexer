"""
Chunker module for Project ACI.

Provides code chunking capabilities with AST-based semantic chunking
and fixed-size fallback for unsupported languages.
"""

import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from aci.core.ast_parser import ASTNode
from aci.core.file_scanner import ScannedFile
from aci.core.tokenizer import TokenizerInterface

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """
    Represents a chunk of code for indexing.

    Attributes:
        chunk_id: Unique identifier for the chunk
        file_path: Path to the source file
        start_line: Start line number (1-based)
        end_line: End line number (1-based, inclusive)
        content: The actual code content
        language: Programming language identifier
        chunk_type: Type of chunk ('function', 'class', 'method', 'fixed')
        metadata: Additional metadata (function_name, parent_class, imports, file_hash)
    """

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    language: str = ""
    chunk_type: str = ""  # 'function', 'class', 'method', 'fixed'
    metadata: dict = field(default_factory=dict)


class ChunkerInterface(ABC):
    """Abstract interface for code chunking operations."""

    @abstractmethod
    def chunk(self, file: ScannedFile, ast_nodes: List[ASTNode]) -> List[CodeChunk]:
        """
        Split a file into code chunks.

        Args:
            file: The scanned file to chunk
            ast_nodes: AST nodes extracted from the file (may be empty)

        Returns:
            List of CodeChunk objects

        Notes:
            - Uses AST nodes for semantic chunking when available
            - Falls back to fixed-size chunking when no AST nodes
            - Splits oversized chunks to fit within token limits
        """
        pass

    @abstractmethod
    def set_max_tokens(self, max_tokens: int) -> None:
        """
        Set the maximum token count per chunk.

        Args:
            max_tokens: Maximum tokens allowed per chunk
        """
        pass


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
    ) -> List[CodeChunk]:
        """
        拆分超大 AST 节点

        Args:
            node: The oversized AST node to split
            max_tokens: Maximum tokens per chunk
            file_path: Path to the source file
            language: Programming language
            base_metadata: Base metadata for chunks

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
            # For methods, include class context
            return f"# Context: class {node.parent_name}\n"
        elif node.node_type == "function":
            # For functions, include function name context
            return f"# Context: function {node.name}\n"
        elif node.node_type == "class":
            # For classes, include class name context
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
                # Reached the end
                break

            # Find the best split point within the valid range
            best_split = self._find_best_split_point(lines, current_start, end_idx)

            if best_split > current_start:
                split_points.append(best_split)
                current_start = best_split
            else:
                # No good split point found, use end_idx
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

        Args:
            lines: List of code lines
            start_idx: Starting line index
            max_tokens: Maximum tokens allowed

        Returns:
            Maximum end index (exclusive)
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

        Args:
            lines: List of code lines
            start_idx: Start of search range
            end_idx: End of search range (exclusive)

        Returns:
            Best split point index
        """
        # Search backwards from end_idx to find best split point
        empty_line_candidates = []
        statement_boundary_candidates = []
        low_indent_candidates = []

        # Get base indentation level
        base_indent = self._get_indentation(lines[start_idx]) if start_idx < len(lines) else 0

        for i in range(end_idx - 1, start_idx, -1):
            if i >= len(lines):
                continue

            line = lines[i]

            # Priority 1: Empty lines
            if not line.strip():
                empty_line_candidates.append(i + 1)  # Split after empty line
                continue

            # Priority 2: Statement boundaries
            if self._is_statement_boundary(line):
                statement_boundary_candidates.append(i)

            # Priority 3: Lines with same or lower indentation as base
            current_indent = self._get_indentation(line)
            if current_indent <= base_indent and line.strip():
                low_indent_candidates.append(i)

        # Return best candidate by priority
        if empty_line_candidates:
            return empty_line_candidates[0]  # Most recent empty line
        if statement_boundary_candidates:
            return statement_boundary_candidates[0]
        if low_indent_candidates:
            return low_indent_candidates[0]

        # Fallback: split at end_idx
        return end_idx

    def _is_statement_boundary(self, line: str) -> bool:
        """
        Check if a line represents a statement boundary.

        Args:
            line: The line to check

        Returns:
            True if the line is a statement boundary
        """
        for pattern in self.STATEMENT_BOUNDARY_PATTERNS:
            if pattern.match(line):
                return True
        return False

    def _get_indentation(self, line: str) -> int:
        """
        Get the indentation level of a line.

        Args:
            line: The line to check

        Returns:
            Number of leading whitespace characters
        """
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
    ) -> List[CodeChunk]:
        """
        Create CodeChunk objects from split points.

        Args:
            lines: List of code lines
            split_points: List of split point indices
            node: Original AST node
            file_path: Path to source file
            language: Programming language
            base_metadata: Base metadata for chunks
            context_prefix: Context prefix to add to continuation chunks
            max_tokens: Maximum tokens per chunk

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        for i, start_idx in enumerate(split_points):
            # Determine end index
            if i + 1 < len(split_points):
                end_idx = split_points[i + 1]
            else:
                end_idx = len(lines)

            if start_idx >= end_idx:
                continue

            # Get chunk content
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = "\n".join(chunk_lines)

            # Add context prefix for continuation chunks (not the first one)
            if i > 0 and context_prefix:
                chunk_content = context_prefix + chunk_content

            # Verify token count (should be within limit, but double-check)
            token_count = self._tokenizer.count_tokens(chunk_content)
            if token_count > max_tokens:
                logger.warning(
                    f"Chunk exceeds token limit ({token_count} > {max_tokens}), "
                    f"this may indicate a very long single line"
                )

            # Build metadata
            metadata = base_metadata.copy()
            metadata["function_name"] = node.name
            if node.parent_name:
                metadata["parent_class"] = node.parent_name
            metadata["is_partial"] = True
            metadata["part_index"] = i
            metadata["total_parts"] = len(split_points)
            if i > 0:
                metadata["has_context_prefix"] = bool(context_prefix)

            # Calculate actual line numbers in the original file
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

        return chunks


class ImportExtractorInterface(ABC):
    """Abstract interface for language-specific import extraction."""

    @abstractmethod
    def extract(self, content: str) -> List[str]:
        """
        Extract import statements from code content.

        Args:
            content: Source code content

        Returns:
            List of import statement strings
        """
        pass


class PythonImportExtractor(ImportExtractorInterface):
    """Import extractor for Python."""

    def extract(self, content: str) -> List[str]:
        imports = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
            elif (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith('"""')
                and not stripped.startswith("'''")
            ):
                if not any(
                    stripped.startswith(kw) for kw in ["import ", "from ", "#", '"""', "'''"]
                ):
                    break
        return imports


class JavaScriptImportExtractor(ImportExtractorInterface):
    """Import extractor for JavaScript/TypeScript."""

    def extract(self, content: str) -> List[str]:
        imports = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or (
                stripped.startswith("const ") and " require(" in stripped
            ):
                imports.append(stripped)
            elif stripped.startswith("export "):
                continue
            elif stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
                if not stripped.startswith("import ") and "require(" not in stripped:
                    break
        return imports


class GoImportExtractor(ImportExtractorInterface):
    """Import extractor for Go."""

    def extract(self, content: str) -> List[str]:
        imports = []
        in_import_block = False
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ("):
                in_import_block = True
                continue
            elif in_import_block:
                if stripped == ")":
                    break
                elif stripped and not stripped.startswith("//"):
                    imports.append(stripped.strip('"'))
            elif stripped.startswith("import "):
                import_path = stripped.replace("import ", "").strip().strip('"')
                imports.append(import_path)
        return imports


class NullImportExtractor(ImportExtractorInterface):
    """Null extractor for unsupported languages."""

    def extract(self, content: str) -> List[str]:
        return []


class ImportExtractorRegistry:
    """
    Registry for language-specific import extractors.

    Follows the Open-Closed Principle: new languages can be added
    without modifying existing code.
    """

    def __init__(self):
        self._extractors: dict[str, ImportExtractorInterface] = {}
        self._null_extractor = NullImportExtractor()

    def register(
        self, language: str, extractor: ImportExtractorInterface
    ) -> "ImportExtractorRegistry":
        """
        Register an import extractor for a language.

        Args:
            language: Language identifier
            extractor: Import extractor instance

        Returns:
            Self for method chaining
        """
        self._extractors[language] = extractor
        return self

    def get(self, language: str) -> ImportExtractorInterface:
        """
        Get the import extractor for a language.

        Args:
            language: Language identifier

        Returns:
            Import extractor (NullImportExtractor if not registered)
        """
        return self._extractors.get(language, self._null_extractor)

    def extract_imports(self, content: str, language: str) -> List[str]:
        """
        Extract imports using the appropriate extractor.

        Args:
            content: Source code content
            language: Language identifier

        Returns:
            List of import statements
        """
        return self.get(language).extract(content)


def _create_default_import_registry() -> ImportExtractorRegistry:
    """Create and configure the default import extractor registry."""
    registry = ImportExtractorRegistry()
    registry.register("python", PythonImportExtractor())
    registry.register("javascript", JavaScriptImportExtractor())
    registry.register("typescript", JavaScriptImportExtractor())
    registry.register("go", GoImportExtractor())
    return registry


# Global default registry
_default_import_registry = _create_default_import_registry()


def get_import_registry() -> ImportExtractorRegistry:
    """Get the global default import extractor registry."""
    return _default_import_registry


class Chunker(ChunkerInterface):
    """
    Concrete implementation of ChunkerInterface.

    Provides code chunking with:
    - AST-based semantic chunking for supported languages
    - Fixed-size chunking as fallback
    - Token limit enforcement with intelligent splitting
    - Metadata extraction (function_name, parent_class, imports)
    """

    def __init__(
        self,
        tokenizer: TokenizerInterface,
        max_tokens: int = 8192,
        fixed_chunk_lines: int = 50,
        overlap_lines: int = 5,
        import_registry: Optional[ImportExtractorRegistry] = None,
        smart_splitter: Optional[SmartChunkSplitter] = None,
    ):
        """
        Initialize the Chunker.

        Args:
            tokenizer: Tokenizer for token counting
            max_tokens: Maximum tokens per chunk (default: 8192)
            fixed_chunk_lines: Lines per chunk for fixed-size chunking (default: 50)
            overlap_lines: Overlap lines between fixed chunks (default: 5)
            import_registry: Registry for import extractors (uses default if None)
            smart_splitter: SmartChunkSplitter for intelligent oversized node splitting
        """
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._fixed_chunk_lines = fixed_chunk_lines
        self._overlap_lines = overlap_lines
        self._import_registry = import_registry or _default_import_registry
        self._smart_splitter = smart_splitter or SmartChunkSplitter(tokenizer)

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the maximum token count per chunk."""
        self._max_tokens = max_tokens

    def chunk(self, file: ScannedFile, ast_nodes: List[ASTNode]) -> List[CodeChunk]:
        """
        Split a file into code chunks.

        Uses AST nodes for semantic chunking when available,
        falls back to fixed-size chunking otherwise.
        """
        # Extract file-level imports using the registry
        imports = self._import_registry.extract_imports(file.content, file.language)

        # Base metadata for all chunks from this file
        base_metadata = {
            "file_hash": file.content_hash,
            "imports": imports,
        }

        if ast_nodes:
            return self._chunk_with_ast(file, ast_nodes, base_metadata)
        else:
            return self._chunk_fixed_size(file, base_metadata)

    def _chunk_with_ast(
        self,
        file: ScannedFile,
        ast_nodes: List[ASTNode],
        base_metadata: dict,
    ) -> List[CodeChunk]:
        """
        Create chunks based on AST nodes.

        Args:
            file: The scanned file
            ast_nodes: List of AST nodes
            base_metadata: Base metadata to include in all chunks

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        for node in ast_nodes:
            # Build metadata for this chunk
            metadata = base_metadata.copy()

            if node.node_type == "function":
                metadata["function_name"] = node.name
            elif node.node_type == "method":
                metadata["function_name"] = node.name
                if node.parent_name:
                    metadata["parent_class"] = node.parent_name
            elif node.node_type == "class":
                metadata["class_name"] = node.name

            if node.docstring:
                metadata["docstring"] = node.docstring

            # Check token count and split if necessary
            token_count = self._tokenizer.count_tokens(node.content)

            if token_count <= self._max_tokens:
                # Node fits within token limit
                chunk = CodeChunk(
                    file_path=str(file.path),
                    start_line=node.start_line,
                    end_line=node.end_line,
                    content=node.content,
                    language=file.language,
                    chunk_type=node.node_type,
                    metadata=metadata,
                )
                chunks.append(chunk)
            else:
                # Node exceeds token limit, use SmartChunkSplitter
                sub_chunks = self._smart_splitter.split_oversized_node(
                    node=node,
                    max_tokens=self._max_tokens,
                    file_path=str(file.path),
                    language=file.language,
                    base_metadata=metadata,
                )
                chunks.extend(sub_chunks)

        return chunks

    def _chunk_fixed_size(
        self,
        file: ScannedFile,
        base_metadata: dict,
    ) -> List[CodeChunk]:
        """
        Create fixed-size chunks for files without AST support.

        Args:
            file: The scanned file
            base_metadata: Base metadata for chunks

        Returns:
            List of CodeChunk objects
        """
        chunks = []
        lines = file.content.split("\n")
        total_lines = len(lines)

        if total_lines == 0:
            return chunks

        start_idx = 0

        while start_idx < total_lines:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self._fixed_chunk_lines, total_lines)

            # Get chunk content
            chunk_lines = lines[start_idx:end_idx]
            chunk_content = "\n".join(chunk_lines)

            # Check token limit and adjust if needed
            while (
                self._tokenizer.count_tokens(chunk_content) > self._max_tokens
                and len(chunk_lines) > 1
            ):
                chunk_lines = chunk_lines[:-1]
                end_idx -= 1
                chunk_content = "\n".join(chunk_lines)

            # Create chunk (line numbers are 1-based)
            chunk = CodeChunk(
                file_path=str(file.path),
                start_line=start_idx + 1,
                end_line=end_idx,
                content=chunk_content,
                language=file.language,
                chunk_type="fixed",
                metadata=base_metadata.copy(),
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            if end_idx >= total_lines:
                break

            # Apply overlap
            start_idx = end_idx - self._overlap_lines
            if start_idx <= chunks[-1].start_line - 1:
                # Prevent infinite loop if overlap is too large
                start_idx = end_idx

        return chunks


def create_chunker(
    tokenizer: Optional[TokenizerInterface] = None,
    max_tokens: int = 8192,
    fixed_chunk_lines: int = 50,
    overlap_lines: int = 5,
) -> Chunker:
    """
    Factory function to create a Chunker instance.

    Args:
        tokenizer: Tokenizer instance (uses default if None)
        max_tokens: Maximum tokens per chunk
        fixed_chunk_lines: Lines per fixed-size chunk
        overlap_lines: Overlap between fixed chunks

    Returns:
        Configured Chunker instance
    """
    if tokenizer is None:
        from aci.core.tokenizer import get_default_tokenizer

        tokenizer = get_default_tokenizer()

    return Chunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        fixed_chunk_lines=fixed_chunk_lines,
        overlap_lines=overlap_lines,
    )
