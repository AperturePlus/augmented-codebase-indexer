"""
Configuration module for Project ACI.

Supports loading from YAML/JSON files with environment variable overrides.
Default values are loaded from defaults.yaml for maintainability.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Path to the default configuration file
_DEFAULTS_CONFIG_PATH = Path(__file__).parent / "defaults.yaml"

# Cache for default values
_defaults_cache: dict[str, Any] | None = None


def _load_defaults() -> dict[str, Any]:
    """Load default configuration values from defaults.yaml."""
    global _defaults_cache
    
    if _defaults_cache is not None:
        return _defaults_cache
    
    if not _DEFAULTS_CONFIG_PATH.exists():
        logger.warning(f"Defaults config not found: {_DEFAULTS_CONFIG_PATH}")
        _defaults_cache = {}
        return _defaults_cache
    
    try:
        content = _DEFAULTS_CONFIG_PATH.read_text(encoding="utf-8")
        _defaults_cache = yaml.safe_load(content) or {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse defaults config: {e}")
        _defaults_cache = {}
    
    return _defaults_cache


def _get_default(section: str, key: str, fallback: Any = None) -> Any:
    """Get a default value from the defaults config."""
    defaults = _load_defaults()
    section_defaults = defaults.get(section, {})
    return section_defaults.get(key, fallback)


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding service."""

    api_key: str = field(default_factory=lambda: _get_default("embedding", "api_key", ""))
    api_url: str = field(
        default_factory=lambda: _get_default(
            "embedding", "api_url", "https://api.openai.com/v1/embeddings"
        )
    )
    model: str = field(
        default_factory=lambda: _get_default("embedding", "model", "text-embedding-3-small")
    )
    batch_size: int = field(default_factory=lambda: _get_default("embedding", "batch_size", 100))
    max_retries: int = field(default_factory=lambda: _get_default("embedding", "max_retries", 3))
    timeout: float = field(default_factory=lambda: _get_default("embedding", "timeout", 30.0))


@dataclass
class VectorStoreConfig:
    """Configuration for the Qdrant vector store."""

    host: str = field(default_factory=lambda: _get_default("vector_store", "host", "localhost"))
    port: int = field(default_factory=lambda: _get_default("vector_store", "port", 6333))
    collection_name: str = field(
        default_factory=lambda: _get_default("vector_store", "collection_name", "aci_codebase")
    )
    vector_size: int = field(
        default_factory=lambda: _get_default("vector_store", "vector_size", 1536)
    )


@dataclass
class IndexingConfig:
    """Configuration for the indexing process."""

    file_extensions: list[str] = field(
        default_factory=lambda: _get_default(
            "indexing", "file_extensions", [".py", ".js", ".ts", ".go"]
        )
    )
    ignore_patterns: list[str] = field(
        default_factory=lambda: _get_default(
            "indexing",
            "ignore_patterns",
            [
                "__pycache__",
                "*.pyc",
                ".git",
                "node_modules",
                ".venv",
                "venv",
                "*.egg-info",
                "dist",
                "build",
                ".tox",
                ".pytest_cache",
            ],
        )
    )
    max_chunk_tokens: int = field(
        default_factory=lambda: _get_default("indexing", "max_chunk_tokens", 8192)
    )
    chunk_overlap_lines: int = field(
        default_factory=lambda: _get_default("indexing", "chunk_overlap_lines", 2)
    )
    max_workers: int = field(default_factory=lambda: _get_default("indexing", "max_workers", 4))


@dataclass
class SearchConfig:
    """Configuration for the search service."""

    default_limit: int = field(
        default_factory=lambda: _get_default("search", "default_limit", 10)
    )
    use_rerank: bool = field(default_factory=lambda: _get_default("search", "use_rerank", False))
    rerank_model: str = field(
        default_factory=lambda: _get_default(
            "search", "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = field(default_factory=lambda: _get_default("logging", "level", "INFO"))
    format: str = field(
        default_factory=lambda: _get_default(
            "logging", "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )


@dataclass
class ACIConfig:
    """Main configuration class for Project ACI."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, path: Path | str) -> "ACIConfig":
        """
        Load configuration from a YAML or JSON file.

        Args:
            path: Path to the configuration file (.yaml, .yml, or .json)

        Returns:
            ACIConfig instance with loaded values

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the file format is unsupported
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content) or {}
        elif path.suffix == ".json":
            data = json.loads(content) if content.strip() else {}
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "ACIConfig":
        """Create ACIConfig from a dictionary."""
        config = cls()

        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        if "vector_store" in data:
            config.vector_store = VectorStoreConfig(**data["vector_store"])
        if "indexing" in data:
            config.indexing = IndexingConfig(**data["indexing"])
        if "search" in data:
            config.search = SearchConfig(**data["search"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])

        return config

    def apply_env_overrides(self) -> "ACIConfig":
        """
        Apply environment variable overrides to the configuration.

        Environment variables follow the pattern: ACI_<SECTION>_<KEY>
        Examples:
            - ACI_EMBEDDING_API_KEY
            - ACI_EMBEDDING_API_URL
            - ACI_VECTOR_STORE_HOST
            - ACI_INDEXING_MAX_WORKERS
            - ACI_LOGGING_LEVEL

        Returns:
            Self with environment overrides applied
        """
        env_mappings = {
            # Embedding config
            "ACI_EMBEDDING_API_KEY": ("embedding", "api_key", str),
            "ACI_EMBEDDING_API_URL": ("embedding", "api_url", str),
            "ACI_EMBEDDING_MODEL": ("embedding", "model", str),
            "ACI_EMBEDDING_BATCH_SIZE": ("embedding", "batch_size", int),
            "ACI_EMBEDDING_MAX_RETRIES": ("embedding", "max_retries", int),
            "ACI_EMBEDDING_TIMEOUT": ("embedding", "timeout", float),
            # Vector store config
            "ACI_VECTOR_STORE_HOST": ("vector_store", "host", str),
            "ACI_VECTOR_STORE_PORT": ("vector_store", "port", int),
            "ACI_VECTOR_STORE_COLLECTION_NAME": ("vector_store", "collection_name", str),
            "ACI_VECTOR_STORE_VECTOR_SIZE": ("vector_store", "vector_size", int),
            # Indexing config
            "ACI_INDEXING_MAX_CHUNK_TOKENS": ("indexing", "max_chunk_tokens", int),
            "ACI_INDEXING_CHUNK_OVERLAP_LINES": ("indexing", "chunk_overlap_lines", int),
            "ACI_INDEXING_MAX_WORKERS": ("indexing", "max_workers", int),
            # Search config
            "ACI_SEARCH_DEFAULT_LIMIT": ("search", "default_limit", int),
            "ACI_SEARCH_USE_RERANK": ("search", "use_rerank", _parse_bool),
            "ACI_SEARCH_RERANK_MODEL": ("search", "rerank_model", str),
            # Logging config
            "ACI_LOGGING_LEVEL": ("logging", "level", str),
        }

        for env_var, (section, key, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                setattr(section_obj, key, converter(value))

        return self

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Serialize configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: Path | str) -> None:
        """
        Save configuration to a file.

        Args:
            path: Path to save the configuration (.yaml, .yml, or .json)

        Raises:
            ValueError: If the file format is unsupported
        """
        path = Path(path)

        if path.suffix in (".yaml", ".yml"):
            content = self.to_yaml()
        elif path.suffix == ".json":
            content = self.to_json()
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _parse_bool(value: str) -> bool:
    """Parse a string to boolean."""
    return value.lower() in ("true", "1", "yes", "on")


def load_config(config_path: Optional[Path | str] = None, apply_env: bool = True) -> ACIConfig:
    """
    Load configuration with optional environment variable overrides.

    Args:
        config_path: Optional path to config file. If None, uses defaults.
        apply_env: Whether to apply environment variable overrides.

    Returns:
        ACIConfig instance
    """
    if config_path:
        config = ACIConfig.from_file(config_path)
    else:
        config = ACIConfig()

    if apply_env:
        config.apply_env_overrides()

    return config
