"""
Summary Generator module for multi-granularity indexing.

Generates natural language summaries from AST information for functions,
classes, and files. These summaries complement code chunks for semantic search
by providing higher-level semantic descriptions.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from aci.core.parsers.base import ASTNode
from aci.core.summary_artifact import ArtifactType, SummaryArtifact
from aci.core.tokenizer import TokenizerInterface, get_default_tokenizer

logger = logging.getLogger(__name__)

# Default maximum tokens for summaries (fits within embedding limits)
DEFAULT_MAX_SUMMARY_TOKENS = 512


class SummaryGeneratorInterface(ABC):
    """
    Abstract interface for generating code summaries.
    
    Implementations generate natural language summaries from AST information
    for functions, classes, and files.
    """

    @abstractmethod
    def generate_function_summary(
        self, node: ASTNode, file_path: str
    ) -> SummaryArtifact:
        """
        Generate summary for a function node.
        
        Args:
            node: AST node representing a function
            file_path: Path to the source file
            
        Returns:
            SummaryArtifact with function summary
        """
        pass

    @abstractmethod
    def generate_class_summary(
        self, node: ASTNode, methods: List[ASTNode], file_path: str
    ) -> SummaryArtifact:
        """
        Generate summary for a class node with its methods.
        
        Args:
            node: AST node representing a class
            methods: List of method AST nodes belonging to the class
            file_path: Path to the source file
            
        Returns:
            SummaryArtifact with class summary
        """
        pass

    @abstractmethod
    def generate_file_summary(
        self,
        file_path: str,
        language: str,
        imports: List[str],
        nodes: List[ASTNode],
    ) -> SummaryArtifact:
        """
        Generate summary for an entire file.
        
        Args:
            file_path: Path to the source file
            language: Programming language identifier
            imports: List of import statements
            nodes: List of top-level AST nodes
            
        Returns:
            SummaryArtifact with file summary
        """
        pass


class SummaryGenerator(SummaryGeneratorInterface):
    """
    Generates natural language summaries from AST information.
    
    Summary format examples:
    - Function: "Function `calculate_total` takes parameters (items: List, tax_rate: float) 
                and returns float. Calculates the total price including tax."
    - Class: "Class `OrderProcessor` extends BaseProcessor. Methods: process_order, 
              validate_items, calculate_shipping. Handles order processing workflow."
    - File: "File `order_service.py` (Python). Imports: typing, dataclasses. 
             Contains: OrderProcessor class, helper functions. Main purpose: Order management."
    """

    def __init__(
        self,
        tokenizer: Optional[TokenizerInterface] = None,
        max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
    ):
        """
        Initialize the SummaryGenerator.
        
        Args:
            tokenizer: Tokenizer for token counting (uses default if None)
            max_summary_tokens: Maximum tokens for generated summaries
        """
        self._tokenizer = tokenizer or get_default_tokenizer()
        self._max_summary_tokens = max_summary_tokens

    def generate_function_summary(
        self, node: ASTNode, file_path: str
    ) -> SummaryArtifact:
        """
        Generate summary for a function node.
        
        Extracts function name, parameters, return type, and docstring
        to create a natural language summary.
        """
        # Extract function information from content
        params = self._extract_parameters(node.content, node.name)
        return_type = self._extract_return_type(node.content)
        
        # Build summary content
        summary_parts = [f"Function `{node.name}`"]
        
        if params:
            params_str = ", ".join(params)
            summary_parts.append(f"takes parameters ({params_str})")
        
        if return_type:
            summary_parts.append(f"and returns {return_type}")
        
        summary_content = " ".join(summary_parts) + "."
        
        # Add docstring if present
        if node.docstring:
            docstring_clean = self._clean_docstring(node.docstring)
            if docstring_clean:
                summary_content += f" {docstring_clean}"
        
        # Truncate if needed
        summary_content = self._truncate_content(summary_content)
        
        # Build metadata
        metadata = {
            "parameters": params,
            "return_type": return_type,
            "is_async": "async " in node.content.split("\n")[0],
        }
        if node.docstring:
            metadata["docstring"] = node.docstring
        
        return SummaryArtifact(
            file_path=file_path,
            artifact_type=ArtifactType.FUNCTION_SUMMARY,
            name=node.name,
            content=summary_content,
            start_line=node.start_line,
            end_line=node.end_line,
            metadata=metadata,
        )

    def generate_class_summary(
        self, node: ASTNode, methods: List[ASTNode], file_path: str
    ) -> SummaryArtifact:
        """
        Generate summary for a class node with its methods.
        
        Extracts class name, base classes, and method names to create
        a natural language summary.
        """
        # Extract class information
        base_classes = self._extract_base_classes(node.content, node.name)
        method_names = [m.name for m in methods]
        
        # Build summary content
        summary_parts = [f"Class `{node.name}`"]
        
        if base_classes:
            bases_str = ", ".join(base_classes)
            summary_parts.append(f"extends {bases_str}")
        
        summary_content = " ".join(summary_parts) + "."
        
        if method_names:
            methods_str = ", ".join(method_names)
            summary_content += f" Methods: {methods_str}."
        
        # Add docstring if present
        if node.docstring:
            docstring_clean = self._clean_docstring(node.docstring)
            if docstring_clean:
                summary_content += f" {docstring_clean}"
        
        # Truncate if needed
        summary_content = self._truncate_content(summary_content)
        
        # Build metadata
        metadata = {
            "base_classes": base_classes,
            "method_names": method_names,
        }
        if node.docstring:
            metadata["docstring"] = node.docstring
        
        return SummaryArtifact(
            file_path=file_path,
            artifact_type=ArtifactType.CLASS_SUMMARY,
            name=node.name,
            content=summary_content,
            start_line=node.start_line,
            end_line=node.end_line,
            metadata=metadata,
        )

    def generate_file_summary(
        self,
        file_path: str,
        language: str,
        imports: List[str],
        nodes: List[ASTNode],
    ) -> SummaryArtifact:
        """
        Generate summary for an entire file.
        
        Lists top-level definitions, imports, and provides an overview
        of the file's purpose.
        """
        import os
        
        file_name = os.path.basename(file_path)
        
        # Categorize nodes
        functions = [n for n in nodes if n.node_type == "function"]
        classes = [n for n in nodes if n.node_type == "class"]
        
        # Build summary content
        summary_parts = [f"File `{file_name}` ({language})."]
        
        # Add imports (simplified)
        if imports:
            import_modules = self._simplify_imports(imports)
            if import_modules:
                imports_str = ", ".join(import_modules[:10])  # Limit to 10
                if len(import_modules) > 10:
                    imports_str += f" and {len(import_modules) - 10} more"
                summary_parts.append(f"Imports: {imports_str}.")
        
        # Add definitions
        definitions = []
        if classes:
            class_names = [c.name for c in classes]
            definitions.append(f"classes: {', '.join(class_names)}")
        if functions:
            func_names = [f.name for f in functions]
            definitions.append(f"functions: {', '.join(func_names)}")
        
        if definitions:
            summary_parts.append(f"Contains {'; '.join(definitions)}.")
        
        summary_content = " ".join(summary_parts)
        
        # Truncate if needed
        summary_content = self._truncate_content(summary_content)
        
        # Build metadata
        metadata = {
            "language": language,
            "imports": imports,
            "function_count": len(functions),
            "class_count": len(classes),
        }
        
        return SummaryArtifact(
            file_path=file_path,
            artifact_type=ArtifactType.FILE_SUMMARY,
            name=file_name,
            content=summary_content,
            start_line=0,
            end_line=0,
            metadata=metadata,
        )

    def _extract_parameters(self, content: str, func_name: str) -> List[str]:
        """Extract parameter list from function definition."""
        import re
        
        # Match function definition line
        # Handles: def func(params), async def func(params), function func(params)
        pattern = rf"(?:async\s+)?(?:def|function)\s+{re.escape(func_name)}\s*\(([^)]*)\)"
        match = re.search(pattern, content)
        
        if not match:
            return []
        
        params_str = match.group(1).strip()
        if not params_str:
            return []
        
        # Split parameters and clean them
        params = []
        for param in params_str.split(","):
            param = param.strip()
            if param and param != "self" and param != "cls":
                # Remove default values for cleaner display
                if "=" in param:
                    param = param.split("=")[0].strip()
                params.append(param)
        
        return params

    def _extract_return_type(self, content: str) -> Optional[str]:
        """Extract return type annotation from function definition."""
        import re
        
        # Match return type annotation: ) -> Type:
        pattern = r"\)\s*->\s*([^:]+):"
        match = re.search(pattern, content.split("\n")[0])
        
        if match:
            return match.group(1).strip()
        return None

    def _extract_base_classes(self, content: str, class_name: str) -> List[str]:
        """Extract base classes from class definition."""
        import re
        
        # Match class definition: class Name(Base1, Base2):
        pattern = rf"class\s+{re.escape(class_name)}\s*\(([^)]+)\)"
        match = re.search(pattern, content)
        
        if not match:
            return []
        
        bases_str = match.group(1).strip()
        if not bases_str:
            return []
        
        # Split and clean base classes
        bases = []
        for base in bases_str.split(","):
            base = base.strip()
            if base:
                # Remove generic parameters for cleaner display
                if "[" in base:
                    base = base.split("[")[0]
                bases.append(base)
        
        return bases

    def _clean_docstring(self, docstring: str) -> str:
        """Clean and normalize docstring for summary."""
        if not docstring:
            return ""
        
        # Remove leading/trailing whitespace and quotes
        cleaned = docstring.strip()
        
        # Remove triple quotes if present
        for quote in ['"""', "'''"]:
            if cleaned.startswith(quote):
                cleaned = cleaned[3:]
            if cleaned.endswith(quote):
                cleaned = cleaned[:-3]
        
        # Get first paragraph (up to first blank line)
        lines = cleaned.strip().split("\n")
        first_para_lines = []
        for line in lines:
            if not line.strip():
                break
            first_para_lines.append(line.strip())
        
        return " ".join(first_para_lines)

    def _simplify_imports(self, imports: List[str]) -> List[str]:
        """Simplify import statements to module names."""
        import re
        
        modules = []
        for imp in imports:
            # Handle "from X import Y" -> X
            match = re.match(r"from\s+(\S+)\s+import", imp)
            if match:
                modules.append(match.group(1).split(".")[0])
                continue
            
            # Handle "import X" -> X
            match = re.match(r"import\s+(\S+)", imp)
            if match:
                modules.append(match.group(1).split(".")[0])
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for m in modules:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        
        return unique

    def _truncate_content(self, content: str) -> str:
        """Truncate content to fit within token limit."""
        token_count = self._tokenizer.count_tokens(content)
        
        if token_count <= self._max_summary_tokens:
            return content
        
        # Truncate and add indicator
        truncated = self._tokenizer.truncate_to_tokens(
            content, self._max_summary_tokens - 15  # Reserve space for indicator
        )
        return truncated + " [truncated]"


def create_summary_generator(
    tokenizer: Optional[TokenizerInterface] = None,
    max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
) -> SummaryGenerator:
    """
    Factory function to create a SummaryGenerator instance.
    
    Args:
        tokenizer: Tokenizer instance (uses default if None)
        max_summary_tokens: Maximum tokens for summaries
        
    Returns:
        Configured SummaryGenerator instance
    """
    return SummaryGenerator(
        tokenizer=tokenizer,
        max_summary_tokens=max_summary_tokens,
    )
