"""Main BranchSelector pipeline orchestrator."""

from dataclasses import dataclass, field
from typing import Any

from chatroutes_autobranch.core.candidate import Candidate, ScoredCandidate
from chatroutes_autobranch.core.protocols import (
    EntropyStopper,
    NoveltyFilter,
    Scorer,
)
from chatroutes_autobranch.beam.selector import BeamSelector
from chatroutes_autobranch.budget.manager import BudgetManager


@dataclass
class SelectionResult:
    """
    Result of a single step() call.

    Attributes:
        kept: Candidates selected after all filters.
        scored: All scored candidates before beam selection.
        after_beam: Candidates after beam selection.
        after_novelty: Candidates after novelty filtering.
        entropy_decision: Entropy stopper decision.
        should_continue: Whether to continue exploration.
        details: Additional metadata for debugging.

    Example:
        >>> result = selector.step(parent, candidates)
        >>> print(f"Kept {len(result.kept)} candidates")
        >>> print(f"Should continue: {result.should_continue}")
    """

    kept: list[ScoredCandidate]
    scored: list[ScoredCandidate]
    after_beam: list[ScoredCandidate]
    after_novelty: list[ScoredCandidate]
    entropy_decision: dict[str, Any]
    should_continue: bool
    details: dict[str, Any] = field(default_factory=dict)


class BranchSelector:
    """
    Main pipeline orchestrator for controlled branching.

    Pipeline order:
        1. Score all candidates using Scorer
        2. Beam selection (top-K by score)
        3. Novelty filtering (diversity enforcement)
        4. Entropy-based stopping (convergence check)
        5. Budget admission check

    Args:
        beam_selector: BeamSelector for top-K selection.
        novelty_filter: Optional NoveltyFilter for diversity.
        entropy_stopper: Optional EntropyStopper for convergence.
        budget_manager: Optional BudgetManager for cost control.

    Usage:
        >>> selector = BranchSelector(
        ...     beam_selector=BeamSelector(scorer=scorer, k=5),
        ...     novelty_filter=cosine_filter,
        ...     entropy_stopper=shannon_stopper,
        ...     budget_manager=budget_mgr
        ... )
        >>> result = selector.step(parent, candidates)

    Thread-safety:
        Create new instance per request/tree. Not thread-safe.
    """

    def __init__(
        self,
        beam_selector: BeamSelector,
        novelty_filter: NoveltyFilter | None = None,
        entropy_stopper: EntropyStopper | None = None,
        budget_manager: BudgetManager | None = None,
    ):
        self.beam_selector = beam_selector
        self.novelty_filter = novelty_filter
        self.entropy_stopper = entropy_stopper
        self.budget_manager = budget_manager

    def step(
        self,
        parent: Candidate,
        candidates: list[Candidate],
    ) -> SelectionResult:
        """
        Execute single selection step.

        Args:
            parent: Parent candidate for context.
            candidates: New candidates to evaluate.

        Returns:
            SelectionResult with kept candidates and metadata.

        Pipeline:
            1. Score all candidates
            2. Beam selection (top-K)
            3. Novelty filtering (if configured)
            4. Entropy check (if configured)
            5. Budget admission (if configured)

        Raises:
            BudgetExceededError: If budget exceeded in strict mode.
        """
        details = {}

        # Step 1: Score all candidates
        scored = self.beam_selector.scorer.score(parent, candidates)
        details["total_candidates"] = len(scored)

        # Step 2: Beam selection (top-K)
        after_beam = self.beam_selector.select(parent, candidates)
        details["after_beam"] = len(after_beam)

        # Step 3: Novelty filtering
        if self.novelty_filter:
            after_novelty = self.novelty_filter.prune(after_beam)
            details["after_novelty"] = len(after_novelty)
        else:
            after_novelty = after_beam
            details["novelty_filter"] = "disabled"

        # Step 4: Entropy-based stopping
        should_continue = True
        entropy_decision = {}

        if self.entropy_stopper:
            decision = self.entropy_stopper.should_continue(after_novelty)
            entropy_decision = {
                "should_continue": decision.should_continue,
                "entropy": decision.entropy,
                "delta_entropy": decision.delta_entropy,
                "details": decision.details,
            }
            should_continue = decision.should_continue
            details["entropy"] = decision.entropy
        else:
            entropy_decision = {"disabled": True}
            details["entropy_stopper"] = "disabled"

        # Step 5: Budget admission check
        if self.budget_manager and after_novelty:
            # Estimate tokens/ms (placeholder - user provides real estimates)
            est_tokens = len(after_novelty) * 100  # Rough estimate
            est_ms = len(after_novelty) * 50  # Rough estimate

            # Check budget (may raise BudgetExceededError in strict mode)
            admitted = self.budget_manager.admit(
                n_new=len(after_novelty),
                est_tokens=est_tokens,
                est_ms=est_ms,
            )

            if not admitted:
                # Soft mode: budget exceeded but no exception
                should_continue = False
                details["budget_exceeded"] = True

            # Record nodes
            self.budget_manager.record_nodes(len(after_novelty))
            details["budget_usage"] = self.budget_manager.usage
        else:
            details["budget_manager"] = "disabled"

        # Determine final kept candidates
        # NOTE: should_continue indicates whether to expand these candidates further,
        # but we still return the filtered candidates. The caller decides whether
        # to add them to the exploration queue based on should_continue.
        kept = after_novelty

        return SelectionResult(
            kept=kept,
            scored=scored,
            after_beam=after_beam,
            after_novelty=after_novelty,
            entropy_decision=entropy_decision,
            should_continue=should_continue,
            details=details,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BranchSelector":
        """
        Create BranchSelector from configuration dict.

        Args:
            config: Configuration dictionary (see config/loader.py).

        Returns:
            Configured BranchSelector instance.

        Example:
            >>> config = load_config("config.yaml")
            >>> selector = BranchSelector.from_config(config)

        Config format:
            beam:
              k: 5
            scorer:
              type: "composite"  # or "static"
              weights: {...}  # for composite
              scores: {...}  # for static
            novelty:
              type: "cosine"  # or "mmr" or null
              threshold: 0.85
              lambda_param: 0.5  # for mmr
            entropy:
              type: "shannon"  # or null
              min_entropy: 0.6
              k_max: 5
            budget:
              max_nodes: 32
              max_tokens: 30000
              max_ms: 12000
              mode: "strict"  # or "soft"
            embedding:
              type: "dummy"  # or "openai", "huggingface"
              dimension: 128  # for dummy
        """
        from chatroutes_autobranch.core.embeddings import DummyEmbeddingProvider
        from chatroutes_autobranch.core.scorer import CompositeScorer, StaticScorer
        from chatroutes_autobranch.novelty.cosine import CosineNoveltyFilter
        from chatroutes_autobranch.novelty.mmr import MMRNoveltyFilter
        from chatroutes_autobranch.entropy.shannon import ShannonEntropyStopper
        from chatroutes_autobranch.budget.manager import Budget, BudgetManager

        # 1. Create embedding provider (if needed for novelty/entropy)
        embedding_provider = None
        if "embedding" in config:
            emb_config = config["embedding"]
            emb_type = emb_config.get("type", "dummy")
            if emb_type == "dummy":
                dimension = emb_config.get("dimension", 1536)
                seed = emb_config.get("seed", 42)
                embedding_provider = DummyEmbeddingProvider(dimension=dimension, seed=seed)
            # TODO: Add OpenAI, HuggingFace providers in future
            else:
                raise ValueError(f"Unsupported embedding type: {emb_type}")

        # 2. Create scorer
        scorer_config = config.get("scorer", {})
        scorer_type = scorer_config.get("type", "composite")

        if scorer_type == "static":
            scores = scorer_config.get("scores", {})
            scorer = StaticScorer(scores)
        elif scorer_type == "composite":
            weights = scorer_config.get("weights", {
                "confidence": 0.3,
                "relevance": 0.25,
                "novelty_parent": 0.2,
                "intent_alignment": 0.15,
                "historical_reward": 0.1,
            })
            scorer = CompositeScorer(
                weights=weights,
                embedding_provider=embedding_provider,
            )
        else:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")

        # 3. Create beam selector
        beam_config = config.get("beam", {})
        k = beam_config.get("k", 5)
        beam_selector = BeamSelector(scorer=scorer, k=k)

        # 4. Create novelty filter (optional)
        novelty_filter = None
        if "novelty" in config and config["novelty"]:
            novelty_config = config["novelty"]
            novelty_type = novelty_config.get("type")

            if novelty_type == "cosine":
                threshold = novelty_config.get("threshold", 0.85)
                novelty_filter = CosineNoveltyFilter(
                    threshold=threshold,
                    embedding_provider=embedding_provider,
                )
            elif novelty_type == "mmr":
                lambda_param = novelty_config.get("lambda_param", 0.5)
                threshold = novelty_config.get("threshold")  # optional
                novelty_filter = MMRNoveltyFilter(
                    lambda_param=lambda_param,
                    threshold=threshold,
                    embedding_provider=embedding_provider,
                )
            elif novelty_type is not None:
                raise ValueError(f"Unsupported novelty type: {novelty_type}")

        # 5. Create entropy stopper (optional)
        entropy_stopper = None
        if "entropy" in config and config["entropy"]:
            entropy_config = config["entropy"]
            entropy_type = entropy_config.get("type")

            if entropy_type == "shannon":
                min_entropy = entropy_config.get("min_entropy", 0.6)
                k_max = entropy_config.get("k_max", 5)
                delta_epsilon = entropy_config.get("delta_epsilon")
                random_seed = entropy_config.get("random_seed", 42)
                entropy_stopper = ShannonEntropyStopper(
                    min_entropy=min_entropy,
                    k_max=k_max,
                    delta_epsilon=delta_epsilon,
                    embedding_provider=embedding_provider,
                    random_seed=random_seed,
                )
            elif entropy_type is not None:
                raise ValueError(f"Unsupported entropy type: {entropy_type}")

        # 6. Create budget manager (optional)
        budget_manager = None
        if "budget" in config and config["budget"]:
            budget_config = config["budget"]
            max_nodes = budget_config.get("max_nodes", 32)
            max_tokens = budget_config.get("max_tokens", 30000)
            max_ms = budget_config.get("max_ms", 12000)
            mode = budget_config.get("mode", "strict")

            budget = Budget(
                max_nodes=max_nodes,
                max_tokens=max_tokens,
                max_ms=max_ms,
            )
            budget_manager = BudgetManager(budget, mode=mode)

        # 7. Create and return BranchSelector
        return cls(
            beam_selector=beam_selector,
            novelty_filter=novelty_filter,
            entropy_stopper=entropy_stopper,
            budget_manager=budget_manager,
        )

    def reset(self) -> None:
        """Reset stateful components for new tree exploration."""
        if self.entropy_stopper:
            self.entropy_stopper.reset()

    def get_state(self) -> dict[str, Any]:
        """Serialize state for checkpointing."""
        state: dict[str, Any] = {}

        if self.entropy_stopper:
            state["entropy_stopper"] = self.entropy_stopper.get_state()

        if self.budget_manager:
            state["budget_manager"] = self.budget_manager.get_state()

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        if self.entropy_stopper and "entropy_stopper" in state:
            self.entropy_stopper.set_state(state["entropy_stopper"])

        if self.budget_manager and "budget_manager" in state:
            self.budget_manager.set_state(state["budget_manager"])
