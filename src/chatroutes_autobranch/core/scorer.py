"""Scorer implementations."""

from chatroutes_autobranch.core.candidate import Candidate, ScoredCandidate
from chatroutes_autobranch.core.protocols import EmbeddingProvider


class StaticScorer:
    """
    Static scorer that returns fixed scores for testing.

    Args:
        scores: Dictionary mapping candidate IDs to scores.

    Example:
        >>> scorer = StaticScorer({"c1": 0.9, "c2": 0.5})
        >>> parent = Candidate(id="root", text="Question")
        >>> candidates = [
        ...     Candidate(id="c1", text="Answer 1"),
        ...     Candidate(id="c2", text="Answer 2"),
        ... ]
        >>> scored = scorer.score(parent, candidates)
        >>> scored[0].score == 0.9
        True
    """

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score(
        self, parent: Candidate, candidates: list[Candidate]
    ) -> list[ScoredCandidate]:
        """Score candidates using fixed scores from dictionary."""
        scored = []
        for c in candidates:
            score = self.scores.get(c.id, 0.5)  # Default to 0.5 if not found
            scored.append(
                ScoredCandidate(id=c.id, text=c.text, meta=c.meta.copy(), score=score)
            )
        return scored


class CompositeScorer:
    """
    Composite scorer combining multiple weighted features.

    Features:
        - confidence: Model certainty (from logprobs)
        - relevance: Semantic similarity to parent
        - novelty_parent: Diversity from parent approach
        - intent_alignment: Match to target intent
        - historical_reward: Learning from past outcomes (optional)

    Args:
        embedding_provider: Provider for computing embeddings.
        weights: Feature weights (auto-normalized to sum to 1.0).
        reward_provider: Optional historical reward provider.

    TODO: Implement full composite scoring logic.
          For now, returns dummy scores based on text length.

    Example:
        >>> from chatroutes_autobranch import DummyEmbeddingProvider
        >>> scorer = CompositeScorer(
        ...     embedding_provider=DummyEmbeddingProvider(),
        ...     weights={"confidence": 0.4, "relevance": 0.3, "novelty_parent": 0.3}
        ... )
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        weights: dict[str, float] | None = None,
        reward_provider: any = None,  # TODO: Type this properly
    ):
        self.embedding_provider = embedding_provider
        self.reward_provider = reward_provider

        # Default weights
        default_weights = {
            "confidence": 0.40,
            "relevance": 0.30,
            "novelty_parent": 0.20,
            "intent_alignment": 0.10,
        }

        # Use provided weights or defaults
        self.weights_raw = weights if weights is not None else default_weights

        # Normalize weights
        weight_sum = sum(w for w in self.weights_raw.values() if w > 0)
        if weight_sum > 0:
            self.weights = {k: v / weight_sum for k, v in self.weights_raw.items()}
        else:
            self.weights = default_weights

    def score(
        self, parent: Candidate, candidates: list[Candidate]
    ) -> list[ScoredCandidate]:
        """
        Score candidates using composite weighted features.

        Algorithm (from chatroutes_autobranch_v1.0.md Section 4.2):
            1. Extract features from candidate metadata
            2. Normalize each feature to [0, 1]
            3. Apply weights and compute weighted sum: score = Σ(weight_i × feature_i)
            4. Clamp final scores to [0, 1]
            5. Return list of ScoredCandidate objects

        Features:
            - confidence: Model certainty from logprobs (default: 0.5)
            - relevance: Cosine similarity to parent, mapped from [-1,1] to [0,1]
            - novelty_parent: 1 - cosine similarity to parent, clamped to [0,1]
            - intent_alignment: Target intent match (default: 0.5)
            - historical_reward: Behavioral telemetry (default: 0.5, optional)

        Edge cases handled:
            - Empty candidates list: returns empty list
            - Missing metadata: uses default values (0.5 for neutral)
            - Invalid scores (NaN/Inf): clamped to 0.0
            - Missing embeddings: handled by embedding provider
        """
        import math
        import numpy as np

        # Edge case: empty candidates
        if not candidates:
            return []

        scored = []

        # Compute embeddings once for all candidates and parent
        # This batches the embedding API call for efficiency
        try:
            parent_embedding = self.embedding_provider.embed([parent.text])[0]
            candidate_texts = [c.text for c in candidates]
            candidate_embeddings = self.embedding_provider.embed(candidate_texts)
        except Exception as e:
            # If embedding fails, fall back to neutral scores
            # Log warning and use default values for relevance/novelty
            parent_embedding = None
            candidate_embeddings = [None] * len(candidates)

        for idx, candidate in enumerate(candidates):
            features = {}

            # 1. CONFIDENCE: Extract from logprobs metadata
            # Default to 0.5 (neutral) if missing
            if "logprobs" in candidate.meta:
                logprob = candidate.meta["logprobs"]
                # Convert logprob to probability using exp()
                # Clamp to reasonable range to avoid overflow/underflow
                logprob = max(min(logprob, 0), -10)  # Clamp to [-10, 0]
                features["confidence"] = math.exp(logprob)
            else:
                features["confidence"] = 0.5

            # 2. RELEVANCE: Cosine similarity to parent
            # Map from [-1, 1] to [0, 1] using: (cosine + 1) / 2
            if parent_embedding is not None and candidate_embeddings[idx] is not None:
                # Compute cosine similarity
                parent_vec = np.array(parent_embedding)
                candidate_vec = np.array(candidate_embeddings[idx])

                # Cosine similarity = dot(a, b) / (norm(a) * norm(b))
                dot_product = np.dot(parent_vec, candidate_vec)
                norm_parent = np.linalg.norm(parent_vec)
                norm_candidate = np.linalg.norm(candidate_vec)

                if norm_parent > 0 and norm_candidate > 0:
                    cosine_sim = dot_product / (norm_parent * norm_candidate)
                    # Map from [-1, 1] to [0, 1]
                    features["relevance"] = (cosine_sim + 1.0) / 2.0
                else:
                    features["relevance"] = 0.5  # Default if zero vectors
            else:
                features["relevance"] = 0.5  # Default if embeddings unavailable

            # 3. NOVELTY_PARENT: 1 - cosine similarity
            # Measures how different the candidate is from parent
            if parent_embedding is not None and candidate_embeddings[idx] is not None:
                parent_vec = np.array(parent_embedding)
                candidate_vec = np.array(candidate_embeddings[idx])

                dot_product = np.dot(parent_vec, candidate_vec)
                norm_parent = np.linalg.norm(parent_vec)
                norm_candidate = np.linalg.norm(candidate_vec)

                if norm_parent > 0 and norm_candidate > 0:
                    cosine_sim = dot_product / (norm_parent * norm_candidate)
                    # Novelty = 1 - similarity
                    features["novelty_parent"] = 1.0 - cosine_sim
                    # Clamp to [0, 1] (cosine can be in [-1, 1])
                    features["novelty_parent"] = max(0.0, min(1.0, features["novelty_parent"]))
                else:
                    features["novelty_parent"] = 0.5
            else:
                features["novelty_parent"] = 0.5

            # 4. INTENT_ALIGNMENT: Check target intent
            # Precedence: candidate.meta["target_intent"] > parent.meta["intent"]
            # Default to 0.5 if missing (as per spec)
            target_intent = candidate.meta.get("target_intent") or parent.meta.get("intent")

            if target_intent:
                # TODO: Implement intent classifier when available
                # For now, use simple heuristic: default to 0.5
                # Real implementation would call intent classifier here
                features["intent_alignment"] = 0.5
            else:
                features["intent_alignment"] = 0.5

            # 5. HISTORICAL_REWARD: Optional behavioral feedback
            # Only included if reward_provider is configured
            if self.reward_provider is not None and "historical_reward" in self.weights:
                try:
                    # Extract context for reward lookup (as per spec Section 4.2.3)
                    context = {
                        "tenant_id": candidate.meta.get("tenant_id")
                        or parent.meta.get("tenant_id"),
                        "intent": parent.meta.get("intent"),
                        "template_id": candidate.meta.get("template_id"),
                        "parent_id": parent.id,
                    }
                    features["historical_reward"] = self.reward_provider.get_reward(
                        candidate, context
                    )
                except Exception:
                    # If reward lookup fails, use neutral default
                    features["historical_reward"] = 0.5

            # COMPUTE WEIGHTED SUM
            # score = Σ(weight_i × feature_i) for features with non-zero weights
            score = 0.0
            for feature_name, weight in self.weights.items():
                if weight > 0 and feature_name in features:
                    feature_value = features[feature_name]
                    # Handle invalid values (NaN, Inf)
                    if not math.isfinite(feature_value):
                        feature_value = 0.0
                    score += weight * feature_value

            # CLAMP FINAL SCORE TO [0, 1]
            score = max(0.0, min(1.0, score))

            # Create scored candidate
            scored.append(
                ScoredCandidate(
                    id=candidate.id, text=candidate.text, meta=candidate.meta.copy(), score=score
                )
            )

        return scored
