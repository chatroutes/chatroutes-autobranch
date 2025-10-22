"""Protocol definitions for pluggable components."""

from typing import Protocol, Any
from dataclasses import dataclass
from chatroutes_autobranch.core.candidate import Candidate, ScoredCandidate


class EmbeddingProvider(Protocol):
    """
    Protocol for embedding text into vector representations.

    Implementations:
        - OpenAIEmbeddingProvider: Uses OpenAI's embedding API
        - HFEmbeddingProvider: Uses HuggingFace sentence-transformers
        - DummyEmbeddingProvider: Returns random embeddings (testing only)
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
            Each vector is a list of floats.

        Example:
            >>> provider = OpenAIEmbeddingProvider()
            >>> embeddings = provider.embed(["Hello", "World"])
            >>> len(embeddings) == 2
            True
            >>> isinstance(embeddings[0], list)
            True
        """
        ...


class Scorer(Protocol):
    """
    Protocol for scoring candidates relative to a parent.

    Implementations:
        - CompositeScorer: Weighted combination of multiple features
        - StaticScorer: Fixed scores (testing only)
    """

    def score(
        self, parent: Candidate, candidates: list[Candidate]
    ) -> list[ScoredCandidate]:
        """
        Score candidates relative to parent context.

        Args:
            parent: The parent candidate (context for scoring).
            candidates: List of child candidates to score.

        Returns:
            List of scored candidates (same length as input candidates).
            Scores are normalized to [0,1].

        Example:
            >>> scorer = CompositeScorer(embedding_provider=provider)
            >>> parent = Candidate(id="root", text="Question")
            >>> candidates = [Candidate(id="c1", text="Answer 1")]
            >>> scored = scorer.score(parent, candidates)
            >>> 0.0 <= scored[0].score <= 1.0
            True
        """
        ...


class NoveltyFilter(Protocol):
    """
    Protocol for pruning similar candidates.

    Implementations:
        - CosineNoveltyFilter: Cosine similarity threshold
        - MMRNoveltyFilter: Maximal Marginal Relevance
    """

    def prune(self, candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
        """
        Prune similar candidates from a score-descending list.

        Args:
            candidates: List of scored candidates, MUST be sorted by
                       score descending (BeamSelector guarantees this).

        Returns:
            Subset of candidates with similar items removed.
            Maintains score-descending order.

        Determinism:
            - If similarity > threshold: keep higher-scored item
            - If scores tied: keep lower lexicographic ID

        Example:
            >>> filter = CosineNoveltyFilter(threshold=0.85)
            >>> kept = filter.prune(scored_candidates)
            >>> len(kept) <= len(scored_candidates)
            True
        """
        ...


@dataclass
class EntropyDecision:
    """
    Decision from entropy stopper about whether to continue branching.

    Attributes:
        should_continue: If True, continue exploring; if False, stop.
        entropy: Normalized entropy value [0,1].
        delta_entropy: Change from previous generation (None if first).
        details: Additional information (clusters, reason for stopping, etc.).
    """

    should_continue: bool
    entropy: float
    delta_entropy: float | None
    details: dict[str, Any]


class EntropyStopper(Protocol):
    """
    Protocol for deciding when to stop tree exploration based on entropy.

    Implementation:
        - ShannonEntropyStopper: Shannon entropy on K-means clusters
    """

    def should_continue(self, kept: list[ScoredCandidate]) -> EntropyDecision:
        """
        Decide whether to continue branching based on diversity.

        Args:
            kept: List of candidates kept after beam and novelty filtering.

        Returns:
            EntropyDecision with continuation decision and metrics.

        Edge cases:
            - len(kept) == 0: should_continue=False, entropy=0.0
            - len(kept) == 1: should_continue=False, entropy=0.0
            - len(kept) == 2: proceed with k=2 clustering

        Example:
            >>> stopper = ShannonEntropyStopper(min_entropy=0.6)
            >>> decision = stopper.should_continue(kept_candidates)
            >>> isinstance(decision.should_continue, bool)
            True
        """
        ...

    def reset(self) -> None:
        """
        Clear entropy history for new tree exploration.

        Called when starting a new tree or resuming from checkpoint.
        """
        ...

    def get_state(self) -> dict:
        """
        Serialize state for checkpointing.

        Returns:
            Dictionary with previous entropy, history, etc.
        """
        ...

    def set_state(self, state: dict) -> None:
        """
        Restore state from checkpoint.

        Args:
            state: Dictionary from get_state().
        """
        ...
