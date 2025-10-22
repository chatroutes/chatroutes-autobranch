"""Beam selector for top-K selection."""

from chatroutes_autobranch.core.candidate import Candidate, ScoredCandidate
from chatroutes_autobranch.core.protocols import Scorer


class BeamSelector:
    """
    Beam search selector - keeps top K candidates by score.

    Implements deterministic tie-breaking:
        1. Primary sort: score descending
        2. Ties: lexicographic ID comparison
        3. Stable sort preserves input order for identical IDs

    Args:
        k: Number of candidates to keep (beam width).
        scorer: Scorer instance for assigning scores.

    Example:
        >>> from chatroutes_autobranch import StaticScorer, Candidate
        >>> scorer = StaticScorer({"c1": 0.9, "c2": 0.5, "c3": 0.7})
        >>> beam = BeamSelector(k=2, scorer=scorer)
        >>> parent = Candidate(id="root", text="Question")
        >>> candidates = [
        ...     Candidate(id="c1", text="A1"),
        ...     Candidate(id="c2", text="A2"),
        ...     Candidate(id="c3", text="A3"),
        ... ]
        >>> kept = beam.select(parent, candidates)
        >>> len(kept) == 2
        True
        >>> kept[0].score >= kept[1].score  # Score descending
        True
    """

    def __init__(self, k: int, scorer: Scorer):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.scorer = scorer

    def select(
        self, parent: Candidate, candidates: list[Candidate]
    ) -> list[ScoredCandidate]:
        """
        Score all candidates and return top K.

        Args:
            parent: Parent candidate for context.
            candidates: List of child candidates to score.

        Returns:
            Top K scored candidates, sorted by score descending.
            If len(candidates) < k, returns all candidates.

        Tie-breaking:
            - Scores are compared with float equality
            - Ties broken by lexicographic ID (string comparison)
            - "c10" < "c2" (string order, not numeric)
            - Recommendation: Use zero-padded IDs for predictable ordering
        """
        if not candidates:
            return []

        # Score all candidates
        scored = self.scorer.score(parent, candidates)

        # Sort by score descending, then by ID lexicographically
        # Python sort is stable, so equal elements maintain original order
        sorted_candidates = sorted(
            scored,
            key=lambda c: (-c.score, c.id),  # Negative score for descending
        )

        # Return top K
        return sorted_candidates[: self.k]
