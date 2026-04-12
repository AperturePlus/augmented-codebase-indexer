"""
Qdrant launcher helper.

Ensures a Qdrant container is running on the expected port. Uses
``docker compose up -d`` with the bundled compose file so that container
lifecycle is managed declaratively rather than via ad-hoc ``docker run``.
"""

import logging
import os
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Compose file shipped with the repository.
_COMPOSE_FILE = Path(__file__).parent.parent.parent.parent / "docker" / "qdrant" / "docker-compose.yaml"


def _is_port_open(host: str, port: int) -> bool:
    """Check if a TCP port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def _is_running_in_container() -> bool:
    """Detect whether the current process is running inside a container."""
    return Path("/.dockerenv").exists() or os.environ.get("container", "") == "docker"


def ensure_qdrant_running(
    host: str = "localhost",
    port: int = 6333,
    container_name: str = "aci-qdrant",
    image: str = "qdrant/qdrant:latest",
    url: str | None = None,
) -> None:
    """
    Ensure a Qdrant instance is reachable.

    If Qdrant is not reachable on a local endpoint, starts it via
    ``docker compose up -d`` using the bundled compose file.
    This is best-effort: if Docker Compose is unavailable, a warning is
    logged and execution continues.

    The ``container_name`` and ``image`` parameters are accepted for
    interface compatibility but are not used — the compose file is the
    single source of truth for those values.
    """
    url = (url or "").strip()
    if not url and host.startswith(("http://", "https://")):
        url = host.strip()

    check_host = host
    check_port = port
    if url:
        parsed = urlparse(url)
        if parsed.hostname:
            check_host = parsed.hostname
        if parsed.port:
            check_port = parsed.port

    if check_host == "0.0.0.0":
        check_host = "localhost"

    if _is_port_open(check_host, check_port):
        return

    is_local = check_host in {"localhost", "127.0.0.1", "::1"}
    if not is_local:
        logger.warning(
            "Qdrant endpoint %s:%s is unreachable; skipping Docker auto-start (non-local)",
            check_host,
            check_port,
        )
        return

    if _is_running_in_container():
        logger.warning(
            "Qdrant endpoint %s:%s is unreachable; skipping Docker auto-start inside container. "
            "Run Qdrant as a separate local container or set ACI_VECTOR_STORE_URL.",
            check_host,
            check_port,
        )
        return

    compose_file = _COMPOSE_FILE.resolve()
    if not compose_file.exists():
        logger.warning(
            "Compose file not found at %s; cannot auto-start Qdrant on %s:%s",
            compose_file,
            check_host,
            check_port,
        )
        return

    try:
        env = {
            **os.environ,
            "ACI_VECTOR_STORE_PORT": str(check_port),
        }
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

        if _is_port_open(check_host, check_port):
            logger.info("Started Qdrant via docker compose on %s:%s", check_host, check_port)
        else:
            logger.warning(
                "Attempted to start Qdrant via docker compose, but %s:%s is still unreachable",
                check_host,
                check_port,
            )
    except FileNotFoundError:
        logger.warning(
            "Docker is not installed or not on PATH; cannot auto-start Qdrant on %s:%s",
            check_host,
            check_port,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to auto-start Qdrant: %s", exc)
