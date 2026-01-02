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
from typing import Any

import yaml
from dotenv import load_dotenv

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
    dimension: int = field(default_factory=lambda: _get_default("embedding", "dimension", 1536))


@dataclass
class VectorStoreConfig:
    """Configuration for the Qdrant vector store."""

    url: str = field(default_factory=lambda: _get_default("vector_store", "url", ""))
    host: str = field(default_factory=lambda: _get_default("vector_store", "host", "localhost"))
    port: int = field(default_factory=lambda: _get_default("vector_store", "port", 6333))
    api_key: str = field(default_factory=lambda: _get_default("vector_store", "api_key", ""))
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
            "indexing",
            "file_extensions",
            [
                # Core languages with Tree-sitter support
                ".py", ".pyw", ".pyi",  # Python
                ".js", ".jsx", ".mjs", ".cjs",  # JavaScript
                ".ts", ".tsx", ".mts", ".cts",  # TypeScript
                ".go",  # Go
                ".java",  # Java
                ".c", ".h",  # C
                ".cpp", ".cc", ".cxx", ".hpp", ".hxx",  # C++
            ],
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

    default_limit: int = field(default_factory=lambda: _get_default("search", "default_limit", 10))
    use_rerank: bool = field(default_factory=lambda: _get_default("search", "use_rerank", False))
    rerank_model: str = field(
        default_factory=lambda: _get_default(
            "search", "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    rerank_api_url: str = field(
        default_factory=lambda: _get_default("search", "rerank_api_url", "")
    )
    rerank_api_key: str = field(
        default_factory=lambda: _get_default("search", "rerank_api_key", "")
    )
    rerank_timeout: float = field(
        default_factory=lambda: _get_default("search", "rerank_timeout", 30.0)
    )
    rerank_endpoint: str = field(
        default_factory=lambda: _get_default("search", "rerank_endpoint", "/v1/rerank")
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
class ServerConfig:
    """Configuration for HTTP server."""

    host: str = field(default_factory=lambda: _get_default("server", "host", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_default("server", "port", 8000))


@dataclass
class ACIConfig:
    """Main configuration class for Project ACI."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

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
            "ACI_EMBEDDING_DIMENSION": ("embedding", "dimension", int),
            # Vector store config
            "ACI_VECTOR_STORE_URL": ("vector_store", "url", str),
            "ACI_VECTOR_STORE_HOST": ("vector_store", "host", str),
            "ACI_VECTOR_STORE_PORT": ("vector_store", "port", int),
            "ACI_VECTOR_STORE_API_KEY": ("vector_store", "api_key", str),
            "ACI_VECTOR_STORE_COLLECTION_NAME": ("vector_store", "collection_name", str),
            "ACI_VECTOR_STORE_VECTOR_SIZE": ("vector_store", "vector_size", int),
            # Indexing config
            "ACI_INDEXING_MAX_CHUNK_TOKENS": ("indexing", "max_chunk_tokens", int),
            "ACI_INDEXING_CHUNK_OVERLAP_LINES": ("indexing", "chunk_overlap_lines", int),
            "ACI_INDEXING_MAX_WORKERS": ("indexing", "max_workers", int),
            "ACI_INDEXING_FILE_EXTENSIONS": (
                "indexing",
                "file_extensions",
                lambda v: [s.strip() for s in v.split(",") if s.strip()],
            ),
            "ACI_INDEXING_IGNORE_PATTERNS": (
                "indexing",
                "ignore_patterns",
                lambda v: [s.strip() for s in v.split(",") if s.strip()],
            ),
            # Search config
            "ACI_SEARCH_DEFAULT_LIMIT": ("search", "default_limit", int),
            "ACI_SEARCH_USE_RERANK": ("search", "use_rerank", _parse_bool),
            "ACI_SEARCH_RERANK_MODEL": ("search", "rerank_model", str),
            "ACI_SEARCH_RERANK_API_URL": ("search", "rerank_api_url", str),
            "ACI_SEARCH_RERANK_API_KEY": ("search", "rerank_api_key", str),
            "ACI_SEARCH_RERANK_TIMEOUT": ("search", "rerank_timeout", float),
            "ACI_SEARCH_RERANK_ENDPOINT": ("search", "rerank_endpoint", str),
            # Logging config
            "ACI_LOGGING_LEVEL": ("logging", "level", str),
            # Server config
            "ACI_SERVER_HOST": ("server", "host", str),
            "ACI_SERVER_PORT": ("server", "port", int),
        }

        for env_var, (section, key, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                setattr(section_obj, key, converter(value))

        return self

    @classmethod
    def from_file(cls, path: Path | str) -> "ACIConfig":
        """
        Load configuration from a YAML or JSON file.

        Args:
            path: Path to the configuration file

        Returns:
            ACIConfig instance populated from the file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        content = path.read_text(encoding="utf-8")
        data = {}

        try:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(content) or {}
            elif path.suffix == ".json":
                data = json.loads(content)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        except Exception as e:
            raise ValueError(f"Failed to parse config file {path}: {e}")

        # Helper to safely create nested config objects
        def create_subconfig(config_cls, section_data):
            if not isinstance(section_data, dict):
                return config_cls()
            # Filter out unknown keys to prevent TypeError
            valid_keys = config_cls.__dataclass_fields__.keys()
            filtered_data = {k: v for k, v in section_data.items() if k in valid_keys}
            return config_cls(**filtered_data)

        return cls(
            embedding=create_subconfig(EmbeddingConfig, data.get("embedding", {})),
            vector_store=create_subconfig(VectorStoreConfig, data.get("vector_store", {})),
            indexing=create_subconfig(IndexingConfig, data.get("indexing", {})),
            search=create_subconfig(SearchConfig, data.get("search", {})),
            logging=create_subconfig(LoggingConfig, data.get("logging", {})),
            server=create_subconfig(ServerConfig, data.get("server", {})),
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Serialize configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict_safe(self) -> dict:
        """
        Convert configuration to a dictionary with sensitive fields redacted.

        This method is safe for logging and debugging purposes as it masks
        API keys, passwords, and other sensitive information.

        Returns:
            Dictionary with sensitive fields replaced with '[REDACTED]'
        """
        config_dict = self.to_dict()

        # Redact sensitive fields in embedding config
        if "embedding" in config_dict and "api_key" in config_dict["embedding"]:
            if config_dict["embedding"]["api_key"]:
                config_dict["embedding"]["api_key"] = "[REDACTED]"

        # Redact sensitive fields in search/rerank config
        if "search" in config_dict and "rerank_api_key" in config_dict["search"]:
            if config_dict["search"]["rerank_api_key"]:
                config_dict["search"]["rerank_api_key"] = "[REDACTED]"

        # Redact sensitive fields in vector store config
        if "vector_store" in config_dict and "api_key" in config_dict["vector_store"]:
            if config_dict["vector_store"]["api_key"]:
                config_dict["vector_store"]["api_key"] = "[REDACTED]"

        return config_dict

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


def load_config(config_path: Path | str | None = None, apply_env: bool = True) -> ACIConfig:
    """
    Load configuration from a file (optional) and environment variables.

    Args:
        config_path: Path to YAML/JSON configuration file.
        apply_env: Whether to apply environment variable overrides (and .env).

    Returns:
        ACIConfig instance

    Notes:
        - Automatically loads a local .env file if present (python-dotenv).
        - Priority: Environment Variables > Config File > Defaults
        - Raises ValueError if required keys (e.g., embedding.api_key) remain empty.
    """
    # Load .env early so os.environ picks it up
    load_dotenv()

    if config_path:
        config = ACIConfig.from_file(config_path)
    else:
        config = ACIConfig()

    if apply_env:
        config.apply_env_overrides()

    # Validate required settings
    if not config.embedding.api_key:
        raise ValueError(
            "Missing embedding API key. Set ACI_EMBEDDING_API_KEY in .env or environment."
        )
    if (
        config.search.use_rerank
        and config.search.rerank_api_url
        and not config.search.rerank_api_key
    ):
        raise ValueError(
            "Rerank is enabled but ACI_SEARCH_RERANK_API_KEY is missing. "
            "Set it in .env/environment, or disable use_rerank."
        )
    if config.embedding.dimension != config.vector_store.vector_size:
        raise ValueError(
            f"Embedding dimension ({config.embedding.dimension}) must match "
            f"vector store size ({config.vector_store.vector_size}). "
            "Set ACI_EMBEDDING_DIMENSION and ACI_VECTOR_STORE_VECTOR_SIZE to the same value."
        )

    return config
