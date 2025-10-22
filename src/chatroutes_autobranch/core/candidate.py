"""Core candidate dataclasses."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """
    Represents a candidate branch in tree exploration.

    Attributes:
        id: Unique identifier for this candidate.
           Recommendation: Use zero-padded or UUID for predictable ordering.
        text: The text content of this candidate (e.g., LLM response).
        meta: Optional metadata dictionary.

    Meta Dictionary Standard Keys:
        - "logprobs": float - Model log-probability for confidence scoring
                             (default: 0.5 if missing)
        - "intent": str - Target intent for alignment scoring
                         (preferred location: parent.meta["intent"])
        - "target_intent": str - Per-candidate intent override
                                (precedence: candidate > parent > config)
        - "template_id": str - Template/pattern ID for historical reward lookup
        - "tenant_id": str - Tenant context for multi-tenant reward tracking
        - "exploratory": bool - Mark as unverified/exploratory output
        - "verified": bool - Mark as fact-checked or human-verified

    Unknown keys are allowed for domain-specific metadata.
    Keys starting with "_" are reserved for internal use.

    Examples:
        >>> c = Candidate(id="c001", text="Explain using analogy")
        >>> c_with_meta = Candidate(
        ...     id="c002",
        ...     text="Start with first principles",
        ...     meta={"logprobs": -0.42, "template_id": "explain_deep"}
        ... )
    """

    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate candidate after initialization."""
        if not self.id:
            raise ValueError("Candidate id cannot be empty")
        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be dict, got {type(self.meta)}")


@dataclass
class ScoredCandidate(Candidate):
    """
    Candidate with an assigned score.

    Extends Candidate with a score field added by Scorer.

    Attributes:
        score: Normalized score in [0,1] from composite scoring.
               Higher scores indicate better candidates.

    Examples:
        >>> scored = ScoredCandidate(
        ...     id="c001",
        ...     text="Use classroom analogy",
        ...     score=0.87,
        ...     meta={"logprobs": -0.3}
        ... )
    """

    score: float = 0.0

    def __post_init__(self) -> None:
        """Validate scored candidate."""
        super().__post_init__()

        # Handle NaN/Inf scores
        import math

        if math.isnan(self.score) or math.isinf(self.score):
            import warnings

            warnings.warn(
                f"Invalid score for candidate {self.id}: {self.score}, setting to 0.0",
                UserWarning,
            )
            self.score = 0.0

        # Clamp to [0, 1] (composite scorer should already do this, but enforce)
        self.score = max(0.0, min(1.0, self.score))
