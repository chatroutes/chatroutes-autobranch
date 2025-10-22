"""MMR (Maximal Marginal Relevance) novelty filtering."""

import numpy as np
from chatroutes_autobranch.core.candidate import ScoredCandidate
from chatroutes_autobranch.core.protocols import EmbeddingProvider


class MMRNoveltyFilter:
    """
    Novelty filter using Maximal Marginal Relevance (MMR).

    Balances relevance (score) and diversity using lambda parameter.

    Args:
        lambda_param: Tradeoff between relevance and diversity.
                     0.0 = pure diversity (ignore scores)
                     1.0 = pure relevance (ignore diversity)
                     0.5 = balanced (default)
        threshold: Optional hard diversity constraint (cosine similarity).
                  If set, removes candidates above threshold after MMR.
        embedding_provider: Provider for computing embeddings.

    Algorithm:
        MMR(c) = λ * score(c) - (1-λ) * max_similarity(c, selected)

    Example:
        >>> from chatroutes_autobranch import DummyEmbeddingProvider
        >>> filter = MMRNoveltyFilter(
        ...     lambda_param=0.5,
        ...     threshold=0.85,
        ...     embedding_provider=DummyEmbeddingProvider()
        ... )
    """

    def __init__(
        self,
        lambda_param: float = 0.5,
        threshold: float | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError(f"lambda_param must be in [0,1], got {lambda_param}")

        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")

        self.lambda_param = lambda_param
        self.threshold = threshold
        self.embedding_provider = embedding_provider

    def prune(self, candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
        """
        Select candidates using MMR algorithm.

        Args:
            candidates: Scored candidates (not required to be sorted for MMR).

        Returns:
            Selected candidates maintaining diversity-relevance balance.

        Algorithm:
            1. Initialize selected = []
            2. While candidates remain:
               a. For each unselected c:
                  MMR(c) = λ * score(c) - (1-λ) * max_sim(c, selected)
               b. Select c* = argmax MMR(c)
               c. Add c* to selected
            3. Optional: apply hard threshold filter
        """
        # Handle edge cases
        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates

        # Step 1: Embed all candidates
        if self.embedding_provider is None:
            # No embeddings available, fallback to score-only selection
            return candidates

        texts = [c.text for c in candidates]
        embeddings = self.embedding_provider.embed(texts)

        # Normalize scores to [0, 1] for MMR calculation
        scores = [c.score for c in candidates]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero if all scores are the same
        if max_score == min_score:
            normalized_scores = [1.0] * len(scores)
        else:
            normalized_scores = [
                (s - min_score) / (max_score - min_score) for s in scores
            ]

        # Step 2: Iteratively select candidates using MMR
        selected_indices = []
        unselected_indices = list(range(len(candidates)))

        while unselected_indices:
            best_mmr = -float('inf')
            best_idx = None

            for idx in unselected_indices:
                # Compute MMR score for this candidate
                relevance_score = normalized_scores[idx]

                # Compute max similarity to already selected candidates
                if not selected_indices:
                    # First candidate: no diversity penalty
                    max_similarity = 0.0
                else:
                    similarities = [
                        self._cosine_similarity(embeddings[idx], embeddings[sel_idx])
                        for sel_idx in selected_indices
                    ]
                    max_similarity = max(similarities)

                # MMR formula: λ * score(c) - (1-λ) * max_similarity(c, selected)
                mmr_score = (
                    self.lambda_param * relevance_score
                    - (1 - self.lambda_param) * max_similarity
                )

                # Select candidate with highest MMR score
                # Tie-breaking: if MMR scores equal, prefer lower index (deterministic)
                if mmr_score > best_mmr or (mmr_score == best_mmr and (best_idx is None or idx < best_idx)):
                    best_mmr = mmr_score
                    best_idx = idx

            # Move selected candidate from unselected to selected
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        # Step 3: Create result list in selection order
        selected = [candidates[i] for i in selected_indices]

        # Step 4: Optional hard threshold filter
        if self.threshold is not None:
            selected = self._apply_threshold_filter(selected, embeddings, selected_indices)

        return selected

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

    def _apply_threshold_filter(
        self,
        candidates: list[ScoredCandidate],
        embeddings: list[list[float]],
        selected_indices: list[int]
    ) -> list[ScoredCandidate]:
        """
        Apply optional hard diversity filter after MMR selection.

        Args:
            candidates: Selected candidates from MMR.
            embeddings: All embeddings (indexed by original candidate position).
            selected_indices: Indices of selected candidates in original list.

        Returns:
            Filtered candidates with similarity > threshold removed.
        """
        if not candidates or self.threshold is None:
            return candidates

        kept = []
        kept_indices = []

        for i, candidate in enumerate(candidates):
            original_idx = selected_indices[i]

            # Check similarity against all previously kept candidates
            if not kept_indices:
                # First candidate always kept
                kept.append(candidate)
                kept_indices.append(original_idx)
            else:
                # Compute max similarity to kept candidates
                similarities = [
                    self._cosine_similarity(embeddings[original_idx], embeddings[kept_idx])
                    for kept_idx in kept_indices
                ]
                max_similarity = max(similarities)

                # Keep if below threshold
                if max_similarity <= self.threshold:
                    kept.append(candidate)
                    kept_indices.append(original_idx)

        return kept
