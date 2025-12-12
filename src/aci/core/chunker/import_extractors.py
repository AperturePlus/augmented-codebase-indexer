"""
Import extractors for different programming languages.

Provides language-specific import statement extraction and a registry
for managing extractors.
"""

from typing import List

from .interfaces import ImportExtractorInterface


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
                    import_path = self._extract_package_path(stripped)
                    if import_path:
                        imports.append(import_path)
            elif stripped.startswith("import "):
                remainder = stripped[7:].strip()  # Remove "import "
                import_path = self._extract_package_path(remainder)
                if import_path:
                    imports.append(import_path)
        return imports

    def _extract_package_path(self, import_spec: str) -> str:
        """
        Extract the package path from an import specification.

        Handles:
        - Simple imports: "fmt" -> fmt
        - Aliased imports: f "fmt" -> fmt
        - Dot imports: . "fmt" -> fmt
        - Blank imports: _ "fmt" -> fmt
        """
        first_quote = import_spec.find('"')
        if first_quote == -1:
            return import_spec.strip()

        last_quote = import_spec.rfind('"')
        if last_quote <= first_quote:
            return ""

        return import_spec[first_quote + 1 : last_quote]


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
