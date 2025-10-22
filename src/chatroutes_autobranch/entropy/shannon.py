"""Shannon entropy-based stopping criterion."""

import numpy as np
from chatroutes_autobranch.core.candidate import ScoredCandidate
from chatroutes_autobranch.core.protocols import EntropyDecision, EmbeddingProvider


class ShannonEntropyStopper:
    """
    Entropy stopper using Shannon entropy on K-means clusters.

    Computes diversity of candidates via clustering, decides when to stop
    based on normalized entropy threshold.

    Args:
        min_entropy: Minimum normalized entropy to continue (0.0-1.0).
                    Default: 0.6
        k_max: Maximum number of clusters for K-means.
              Actual k = min(len(candidates), k_max).
              Default: 5
        delta_epsilon: Optional threshold for entropy change.
                      Stop if |Δentropy| < epsilon.
                      Default: None (disabled)
        embedding_provider: Provider for computing embeddings.
        random_seed: Seed for K-means reproducibility.
                    Default: 42

    Example:
        >>> from chatroutes_autobranch import DummyEmbeddingProvider
        >>> stopper = ShannonEntropyStopper(
        ...     min_entropy=0.6,
        ...     k_max=5,
        ...     embedding_provider=DummyEmbeddingProvider()
        ... )
    """

    def __init__(
        self,
        min_entropy: float = 0.6,
        k_max: int = 5,
        delta_epsilon: float | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        random_seed: int = 42,
    ):
        if not 0.0 <= min_entropy <= 1.0:
            raise ValueError(f"min_entropy must be in [0,1], got {min_entropy}")

        if k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {k_max}")

        self.min_entropy = min_entropy
        self.k_max = k_max
        self.delta_epsilon = delta_epsilon
        self.embedding_provider = embedding_provider
        self.random_seed = random_seed

        # State (for stateful entropy tracking)
        self._previous_entropy: float | None = None

    def should_continue(self, kept: list[ScoredCandidate]) -> EntropyDecision:
        """
        Decide whether to continue based on diversity.

        Computes Shannon entropy on K-means clusters and compares to threshold.

        Args:
            kept: Candidates after beam and novelty filtering.

        Returns:
            EntropyDecision with continuation flag and metrics.

        Algorithm:
            1. Embed all candidates using embedding_provider
            2. Run K-means clustering with k = min(len(kept), k_max)
            3. Count members per cluster
            4. Compute Shannon entropy: H = -Σ(p_i × log2(p_i))
            5. Normalize by log2(k) to get entropy in [0, 1]
            6. Compare to min_entropy threshold
            7. Optionally check delta_entropy against delta_epsilon

        Edge cases:
            - len(kept) == 0: return False, entropy=0.0, reason="no_candidates"
            - len(kept) == 1: return False, entropy=0.0, reason="singleton"
            - len(kept) == 2: proceed with k=2

        Stopping conditions:
            - entropy < min_entropy (low diversity)
            - |Δentropy| < delta_epsilon (if configured, entropy change too small)
        """
        # Edge case: no candidates
        if not kept:
            return EntropyDecision(
                should_continue=False,
                entropy=0.0,
                delta_entropy=None,
                details={"reason": "no_candidates_remaining", "clusters": 0},
            )

        # Edge case: single candidate
        if len(kept) == 1:
            return EntropyDecision(
                should_continue=False,
                entropy=0.0,
                delta_entropy=None,
                details={"reason": "singleton", "clusters": 1},
            )

        # Implementation of Shannon entropy computation
        # Step 1: Get embeddings for all candidates
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required for entropy computation")

        texts = [c.text for c in kept]
        embeddings = self.embedding_provider.embed(texts)
        embeddings_array = np.array(embeddings)

        # Step 2: Run K-means clustering with k = min(len(kept), k_max)
        from sklearn.cluster import KMeans

        k = min(len(kept), self.k_max)
        kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        # Step 3: Count members per cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        # Handle edge case: empty clusters (shouldn't happen with proper K-means, but be safe)
        # Filter out any zero counts if they exist
        counts = counts[counts > 0]

        # Step 4: Normalize to probabilities
        total = counts.sum()
        probabilities = counts / total

        # Step 5: Compute Shannon entropy: H = -Σ(p_i × log2(p_i))
        # Filter out zero probabilities to avoid log(0)
        entropy_raw = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Step 6: Normalize by log2(k) to get entropy in [0, 1]
        if k > 1:
            entropy_normalized = entropy_raw / np.log2(k)
        else:
            entropy_normalized = 0.0

        # Ensure entropy is in valid range [0, 1]
        entropy_normalized = np.clip(entropy_normalized, 0.0, 1.0)

        # Step 7: Compute delta entropy if we have previous entropy
        delta_entropy = None
        if self._previous_entropy is not None:
            delta_entropy = abs(entropy_normalized - self._previous_entropy)

        # Step 8: Determine if we should continue
        should_continue = True
        reason = None

        # Check main entropy threshold
        if entropy_normalized < self.min_entropy:
            should_continue = False
            reason = "low_entropy"

        # Check delta_epsilon if configured
        if should_continue and self.delta_epsilon is not None and delta_entropy is not None:
            if delta_entropy < self.delta_epsilon:
                should_continue = False
                reason = "low_delta_entropy"

        # Update state for next iteration
        self._previous_entropy = entropy_normalized

        return EntropyDecision(
            should_continue=should_continue,
            entropy=float(entropy_normalized),
            delta_entropy=float(delta_entropy) if delta_entropy is not None else None,
            details={
                "clusters": k,
                "cluster_sizes": counts.tolist(),
                "reason": reason,
            },
        )

    def reset(self) -> None:
        """Clear entropy history for new tree."""
        self._previous_entropy = None

    def get_state(self) -> dict:
        """Serialize state for checkpointing."""
        return {"previous_entropy": self._previous_entropy}

    def set_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self._previous_entropy = state.get("previous_entropy")
