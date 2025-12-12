"""
Data models and constants for the file scanner module.
"""

from dataclasses import dataclass
from pathlib import Path

# Sensitive patterns that are ALWAYS excluded regardless of user config
# These patterns protect credentials, keys, and private data from being indexed
SENSITIVE_DENYLIST: frozenset[str] = frozenset([
    # SSH and GPG directories
    ".ssh",
    ".gnupg",
    # SSH key files
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "id_ed25519.pub",
    "id_ecdsa",
    "id_ecdsa.pub",
    "id_dsa",
    "id_dsa.pub",
    # Certificates and keys (glob patterns)
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.crt",
    "*.keystore",
    # Environment files
    ".env",
    ".env.*",
    ".env.local",
    ".env.production",
    ".env.development",
    # Other sensitive files
    ".netrc",
    ".npmrc",
    ".pypirc",
])


@dataclass
class ScannedFile:
    """
    Represents a scanned file with its metadata and content.

    Attributes:
        path: Absolute path to the file
        content: File content as UTF-8 string
        language: Detected language identifier ('python', 'javascript', 'typescript', 'go', 'unknown')
        size_bytes: File size in bytes
        modified_time: File modification timestamp (Unix epoch)
        content_hash: SHA-256 hash of the file content
    """

    path: Path
    content: str
    language: str
    size_bytes: int
    modified_time: float
    content_hash: str
