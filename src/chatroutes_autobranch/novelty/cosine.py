"""Cosine similarity-based novelty filtering."""

import numpy as np
from chatroutes_autobranch.core.candidate import ScoredCandidate
from chatroutes_autobranch.core.protocols import EmbeddingProvider


class CosineNoveltyFilter:
    """
    Novelty filter using cosine similarity threshold.

    Prunes candidates that are too similar (cosine similarity > threshold).
    Processes in score-descending order to keep higher-scored items.

    Args:
        threshold: Cosine similarity threshold (0.0-1.0).
                  Higher values = stricter pruning.
                  Default: 0.85
        embedding_provider: Provider for computing embeddings.

    Deterministic tie-breaking:
        - If similarity > threshold: keep higher-scored candidate
        - If scores equal: keep lower lexicographic ID

    Example:
        >>> from chatroutes_autobranch import DummyEmbeddingProvider, ScoredCandidate
        >>> filter = CosineNoveltyFilter(
        ...     threshold=0.85,
        ...     embedding_provider=DummyEmbeddingProvider()
        ... )
        >>> candidates = [
        ...     ScoredCandidate(id="c1", text="Hello world", score=0.9),
        ...     ScoredCandidate(id="c2", text="Hello there", score=0.8),
        ... ]
        >>> kept = filter.prune(candidates)
    """

    def __init__(
        self, threshold: float = 0.85, embedding_provider: EmbeddingProvider | None = None
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")

        self.threshold = threshold
        self.embedding_provider = embedding_provider

    def prune(self, candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
        """
        Prune similar candidates from score-descending list.

        Args:
            candidates: MUST be sorted by score descending
                       (BeamSelector guarantees this).

        Returns:
            Subset with similar items removed, maintaining order.

        Algorithm:
            1. Process candidates in order (highest score first)
            2. For each candidate:
               - Compute similarity to all kept candidates
               - If max_similarity > threshold: prune this candidate
               - Otherwise: keep it
            3. Return kept candidates
        """
        # Edge cases: empty or single candidate
        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates

        # Check for embedding provider
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required for cosine novelty filtering")

        # Step 1: Get embeddings for all candidates in one batch
        texts = [c.text for c in candidates]
        embeddings = self.embedding_provider.embed(texts)

        # Step 2: Greedy selection based on cosine similarity threshold
        kept = []
        kept_embeddings = []

        for i, candidate in enumerate(candidates):
            # First candidate is always kept
            if not kept:
                kept.append(candidate)
                kept_embeddings.append(embeddings[i])
                continue

            # Compute similarity to all previously kept candidates
            similarities = [
                self._cosine_similarity(embeddings[i], kept_emb)
                for kept_emb in kept_embeddings
            ]

            # Get maximum similarity to any kept candidate
            max_similarity = max(similarities)

            # Keep candidate only if it's sufficiently different from all kept ones
            if max_similarity <= self.threshold:
                kept.append(candidate)
                kept_embeddings.append(embeddings[i])
            # Otherwise, prune this candidate (too similar)

        return kept

    def _cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity in [-1, 1].
            Typically in [0, 1] for normalized embeddings.

        Formula:
            cos_sim = dot(a, b) / (norm(a) * norm(b))
        """
        # Convert to numpy arrays
        a = np.array(embedding1)
        b = np.array(embedding2)

        # Compute dot product
        dot_product = np.dot(a, b)

        # Compute norms
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
