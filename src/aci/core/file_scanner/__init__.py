"""
FileScanner module for Project ACI.

Provides recursive directory scanning with file extension filtering,
gitignore pattern support, and content hashing.
"""

from .interfaces import FileScannerInterface
from .language_registry import LanguageRegistry, get_default_registry
from .models import SENSITIVE_DENYLIST, ScannedFile
from .scanner import FileScanner

__all__ = [
    # Main classes
    "FileScanner",
    "FileScannerInterface",
    "ScannedFile",
    # Language registry
    "LanguageRegistry",
    "get_default_registry",
    # Constants
    "SENSITIVE_DENYLIST",
]

# Legacy compatibility - maps to default registry
EXTENSION_TO_LANGUAGE: dict[str, str] = get_default_registry()._extension_to_language
