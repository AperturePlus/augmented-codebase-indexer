"""
Qdrant launcher helper.

Ensures a Qdrant container is running on the expected port. Attempts to
start a local Docker container if Qdrant is not reachable.
"""

import logging
import socket
import subprocess

logger = logging.getLogger(__name__)


def _is_port_open(host: str, port: int) -> bool:
    """Check if a TCP port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def ensure_qdrant_running(
    host: str = "localhost",
    port: int = 6333,
    container_name: str = "aci-qdrant",
    image: str = "qdrant/qdrant:latest",
) -> None:
    """
    Ensure a Qdrant instance is reachable. If not, try to start a Docker container.

    This is best-effort: if Docker is unavailable, we log a warning and continue.
    """
    if _is_port_open(host, port):
        return

    try:
        # Check if the container already exists (including stopped containers)
        # Use -aq to include all containers, not just running ones
        inspect = subprocess.run(
            ["docker", "ps", "-aq", "-f", f"name=^{container_name}$"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if inspect.returncode == 0 and inspect.stdout.strip():
            # Container exists (running or stopped); try to start it
            subprocess.run(
                ["docker", "start", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        else:
            # Container doesn't exist; run a new one
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "-p",
                    f"{port}:6333",
                    image,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )

        if _is_port_open(host, port):
            logger.info("Started Qdrant container on %s:%s", host, port)
        else:
            logger.warning(
                "Attempted to start Qdrant container, but %s:%s is still unreachable",
                host,
                port,
            )
    except FileNotFoundError:
        logger.warning(
            "Docker is not installed or not on PATH; cannot auto-start Qdrant on %s:%s",
            host,
            port,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to auto-start Qdrant: %s", exc)
