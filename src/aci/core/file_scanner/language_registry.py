"""
Language registry for mapping file extensions to programming languages.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default path to the languages configuration file
_DEFAULT_LANGUAGES_CONFIG = Path(__file__).parent.parent / "languages.yaml"


class LanguageRegistry:
    """
    Extensible registry for mapping file extensions to programming languages.

    Supports loading from YAML configuration and runtime registration of new
    languages and their extensions, making it easy to add support for additional
    languages without modifying the core scanner code.

    Example:
        >>> registry = LanguageRegistry()
        >>> registry.register("elixir", [".ex", ".exs"])
        >>> registry.detect(".ex")
        'elixir'

        >>> # Load from custom config
        >>> registry = LanguageRegistry.from_yaml("custom_languages.yaml")
    """

    def __init__(self, load_defaults: bool = True):
        """
        Initialize the language registry.

        Args:
            load_defaults: If True, load default language mappings from languages.yaml.
        """
        self._extension_to_language: dict[str, str] = {}
        self._language_to_extensions: dict[str, set[str]] = {}

        if load_defaults:
            self._load_from_yaml(_DEFAULT_LANGUAGES_CONFIG)

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "LanguageRegistry":
        """
        Create a LanguageRegistry from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            LanguageRegistry instance with loaded mappings

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config file format is invalid
        """
        registry = cls(load_defaults=False)
        registry._load_from_yaml(Path(config_path))
        return registry

    def _load_from_yaml(self, config_path: Path) -> None:
        """
        Load language mappings from a YAML file.

        Expected format:
            language_name:
              - .ext1
              - .ext2
        """
        if not config_path.exists():
            logger.warning(f"Languages config not found: {config_path}, using empty registry")
            return

        try:
            content = config_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)

            if data is None:
                return

            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid languages config format: expected dict, got {type(data)}"
                )

            for language, extensions in data.items():
                if not isinstance(extensions, list):
                    logger.warning(
                        f"Invalid extensions for {language}: expected list, got {type(extensions)}"
                    )
                    continue
                for ext in extensions:
                    self._add_mapping(str(ext), str(language))

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse languages config: {e}")
            raise ValueError(f"Invalid YAML in languages config: {e}") from e

    def _add_mapping(self, extension: str, language: str) -> None:
        """Internal method to add a single extension-language mapping."""
        ext_lower = extension.lower()
        self._extension_to_language[ext_lower] = language

        if language not in self._language_to_extensions:
            self._language_to_extensions[language] = set()
        self._language_to_extensions[language].add(ext_lower)

    def register(self, language: str, extensions: list[str]) -> "LanguageRegistry":
        """
        Register a new language with its file extensions.

        Args:
            language: Language identifier (e.g., 'rust', 'ruby')
            extensions: List of file extensions including the dot (e.g., ['.rs'])

        Returns:
            Self for method chaining
        """
        for ext in extensions:
            self._add_mapping(ext, language)
        return self

    def unregister(self, language: str) -> "LanguageRegistry":
        """
        Remove a language and all its extensions from the registry.

        Args:
            language: Language identifier to remove

        Returns:
            Self for method chaining
        """
        if language in self._language_to_extensions:
            for ext in self._language_to_extensions[language]:
                self._extension_to_language.pop(ext, None)
            del self._language_to_extensions[language]
        return self

    def detect(self, extension: str) -> str:
        """
        Detect language from file extension.

        Args:
            extension: File extension including the dot (e.g., '.py')

        Returns:
            Language identifier or 'unknown' if not recognized
        """
        return self._extension_to_language.get(extension.lower(), "unknown")

    def detect_from_path(self, file_path: Path) -> str:
        """
        Detect language from a file path.

        Args:
            file_path: Path to the file

        Returns:
            Language identifier or 'unknown' if not recognized
        """
        return self.detect(file_path.suffix)

    def get_extensions(self, language: str) -> set[str]:
        """
        Get all registered extensions for a language.

        Args:
            language: Language identifier

        Returns:
            Set of extensions (empty if language not registered)
        """
        return self._language_to_extensions.get(language, set()).copy()

    def get_all_extensions(self) -> set[str]:
        """Get all registered file extensions."""
        return set(self._extension_to_language.keys())

    def get_all_languages(self) -> set[str]:
        """Get all registered language identifiers."""
        return set(self._language_to_extensions.keys())

    def is_supported(self, extension: str) -> bool:
        """Check if an extension is registered."""
        return extension.lower() in self._extension_to_language


# Global default registry instance
_default_registry = LanguageRegistry()


def get_default_registry() -> LanguageRegistry:
    """Get the global default language registry."""
    return _default_registry
