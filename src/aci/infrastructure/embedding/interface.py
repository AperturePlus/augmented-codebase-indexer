"""Abstract interface for embedding clients."""

from abc import ABC, abstractmethod


class EmbeddingClientInterface(ABC):
    """Abstract interface for embedding clients."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        pass
