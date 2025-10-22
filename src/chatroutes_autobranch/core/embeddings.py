"""Embedding provider implementations."""

import random
from typing import Optional


class DummyEmbeddingProvider:
    """
    Dummy embedding provider for testing and offline development.

    Returns random or fixed embeddings without requiring external APIs.

    Args:
        dimension: Embedding dimension (default: 1536 to match OpenAI).
        seed: Random seed for reproducibility (default: None).

    Example:
        >>> provider = DummyEmbeddingProvider(dimension=128, seed=42)
        >>> embeddings = provider.embed(["Hello", "World"])
        >>> len(embeddings) == 2
        True
        >>> len(embeddings[0]) == 128
        True
    """

    def __init__(self, dimension: int = 1536, seed: Optional[int] = None):
        self.dimension = dimension
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate random embeddings for texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of random embedding vectors.
        """
        embeddings = []
        for text in texts:
            # Use text hash for deterministic randomness if seed is set
            if self.seed is not None:
                rng = random.Random(hash(text) + self.seed)
                embedding = [rng.random() for _ in range(self.dimension)]
            else:
                embedding = [random.random() for _ in range(self.dimension)]

            # Normalize to unit vector
            norm = sum(x**2 for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings.append(embedding)

        return embeddings
