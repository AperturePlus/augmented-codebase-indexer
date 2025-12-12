"""
Main Chunker implementation.

Provides code chunking with AST-based semantic chunking
and fixed-size fallback for unsupported languages.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from aci.core.ast_parser import ASTNode
from aci.core.docstring_formatter import DocstringFormatter
from aci.core.file_scanner import ScannedFile
from aci.core.summary_artifact import SummaryArtifact
from aci.core.tokenizer import TokenizerInterface

from .import_extractors import ImportExtractorRegistry, get_import_registry
from .interfaces import ChunkerInterface
from .models import ChunkingResult, CodeChunk
from .smart_splitter import SmartChunkSplitter

if TYPE_CHECKING:
    from aci.core.summary_generator import SummaryGeneratorInterface

logger = logging.getLogger(__name__)


class Chunker(ChunkerInterface):
    """
    Concrete implementation of ChunkerInterface.

    Provides code chunking with:
    - AST-based semantic chunking for supported languages
    - Fixed-size chunking as fallback
    - Token limit enforcement with intelligent splitting
    - Metadata extraction (function_name, parent_class, imports)
    - Summary generation for functions, classes, and files (when SummaryGenerator provided)
    """

    def __init__(
        self,
        tokenizer: TokenizerInterface,
        max_tokens: int = 8192,
        fixed_chunk_lines: int = 50,
        overlap_lines: int = 5,
        import_registry: Optional[ImportExtractorRegistry] = None,
        smart_splitter: Optional[SmartChunkSplitter] = None,
        docstring_formatter: Optional[DocstringFormatter] = None,
        summary_generator: Optional["SummaryGeneratorInterface"] = None,
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
            docstring_formatter: Formatter for docstrings (uses default if None)
            summary_generator: Generator for summary artifacts (optional)
        """
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._fixed_chunk_lines = fixed_chunk_lines
        self._overlap_lines = overlap_lines
        self._import_registry = import_registry or get_import_registry()
        self._smart_splitter = smart_splitter or SmartChunkSplitter(tokenizer)
        self._doc_formatter = docstring_formatter or DocstringFormatter()
        self._summary_generator = summary_generator

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the maximum token count per chunk."""
        self._max_tokens = max_tokens

    def chunk(self, file: ScannedFile, ast_nodes: List[ASTNode]) -> ChunkingResult:
        """
        Split a file into code chunks and generate summaries.

        Uses AST nodes for semantic chunking when available,
        falls back to fixed-size chunking otherwise.
        Generates function/class/file summaries when SummaryGenerator is available.
        """
        imports = self._import_registry.extract_imports(file.content, file.language)

        base_metadata = {
            "file_hash": file.content_hash,
            "imports": imports,
            "language": file.language,
        }

        if ast_nodes:
            chunks, summaries = self._chunk_with_ast(file, ast_nodes, base_metadata, imports)
        else:
            chunks = self._chunk_fixed_size(file, base_metadata)
            summaries = []
            if self._summary_generator:
                try:
                    file_summary = self._summary_generator.generate_file_summary(
                        file_path=str(file.path),
                        language=file.language,
                        imports=imports,
                        nodes=[],
                    )
                    summaries.append(file_summary)
                except Exception as e:
                    logger.warning(f"Failed to generate file summary for {file.path}: {e}")

        return ChunkingResult(chunks=chunks, summaries=summaries)

    def _chunk_with_ast(
        self,
        file: ScannedFile,
        ast_nodes: List[ASTNode],
        base_metadata: dict,
        imports: List[str],
    ) -> tuple[List[CodeChunk], List[SummaryArtifact]]:
        """
        Create chunks based on AST nodes and generate summaries.

        Args:
            file: The scanned file
            ast_nodes: List of AST nodes
            base_metadata: Base metadata to include in all chunks
            imports: List of import statements for file summary

        Returns:
            Tuple of (chunks, summaries)
        """
        chunks = []
        summaries = []

        # Group methods by parent class for class summary generation
        class_methods: dict[str, List[ASTNode]] = {}
        for node in ast_nodes:
            if node.node_type == "method" and node.parent_name:
                if node.parent_name not in class_methods:
                    class_methods[node.parent_name] = []
                class_methods[node.parent_name].append(node)

        for node in ast_nodes:
            metadata = base_metadata.copy()

            if node.node_type == "function":
                metadata["function_name"] = node.name
                if self._summary_generator:
                    try:
                        summary = self._summary_generator.generate_function_summary(
                            node, str(file.path)
                        )
                        summaries.append(summary)
                    except Exception as e:
                        logger.warning(f"Failed to generate function summary for {node.name}: {e}")
            elif node.node_type == "method":
                metadata["function_name"] = node.name
                if node.parent_name:
                    metadata["parent_class"] = node.parent_name
                if self._summary_generator:
                    try:
                        summary = self._summary_generator.generate_function_summary(
                            node, str(file.path)
                        )
                        summaries.append(summary)
                    except Exception as e:
                        logger.warning(f"Failed to generate method summary for {node.name}: {e}")
            elif node.node_type == "class":
                metadata["class_name"] = node.name
                if self._summary_generator:
                    try:
                        methods = class_methods.get(node.name, [])
                        summary = self._summary_generator.generate_class_summary(
                            node, methods, str(file.path)
                        )
                        summaries.append(summary)
                    except Exception as e:
                        logger.warning(f"Failed to generate class summary for {node.name}: {e}")

            formatted_docstring = None
            docstring_prefix = ""
            if node.docstring:
                formatted_docstring = self._doc_formatter.normalize(
                    node.docstring, file.language
                )
                metadata["docstring"] = formatted_docstring
                docstring_prefix = (
                    f"{formatted_docstring}{self._doc_formatter.DELIMITER}"
                    if formatted_docstring
                    else ""
                )

            content_with_doc = (
                f"{docstring_prefix}{node.content}" if docstring_prefix else node.content
            )
            token_count = self._tokenizer.count_tokens(content_with_doc)

            if token_count <= self._max_tokens:
                chunk = CodeChunk(
                    file_path=str(file.path),
                    start_line=node.start_line,
                    end_line=node.end_line,
                    content=content_with_doc,
                    language=file.language,
                    chunk_type=node.node_type,
                    metadata=metadata,
                )
                chunks.append(chunk)
            else:
                sub_chunks = self._smart_splitter.split_oversized_node(
                    node=node,
                    max_tokens=self._max_tokens,
                    file_path=str(file.path),
                    language=file.language,
                    base_metadata=metadata,
                    docstring_prefix=docstring_prefix,
                )
                chunks.extend(sub_chunks)

        # Generate file summary
        if self._summary_generator:
            try:
                file_summary = self._summary_generator.generate_file_summary(
                    file_path=str(file.path),
                    language=file.language,
                    imports=imports,
                    nodes=ast_nodes,
                )
                summaries.append(file_summary)
            except Exception as e:
                logger.warning(f"Failed to generate file summary for {file.path}: {e}")

        return chunks, summaries

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
            end_idx = min(start_idx + self._fixed_chunk_lines, total_lines)

            chunk_lines = lines[start_idx:end_idx]
            chunk_content = "\n".join(chunk_lines)

            while (
                self._tokenizer.count_tokens(chunk_content) > self._max_tokens
                and len(chunk_lines) > 1
            ):
                chunk_lines = chunk_lines[:-1]
                end_idx -= 1
                chunk_content = "\n".join(chunk_lines)

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

            if end_idx >= total_lines:
                break

            start_idx = end_idx - self._overlap_lines
            if start_idx <= chunks[-1].start_line - 1:
                start_idx = end_idx

        return chunks


def create_chunker(
    tokenizer: Optional[TokenizerInterface] = None,
    max_tokens: int = 8192,
    fixed_chunk_lines: int = 50,
    overlap_lines: int = 5,
    summary_generator: Optional["SummaryGeneratorInterface"] = None,
) -> Chunker:
    """
    Factory function to create a Chunker instance.

    Args:
        tokenizer: Tokenizer instance (uses default if None)
        max_tokens: Maximum tokens per chunk
        fixed_chunk_lines: Lines per fixed-size chunk
        overlap_lines: Overlap between fixed chunks
        summary_generator: Generator for summary artifacts (optional)

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
        summary_generator=summary_generator,
    )
