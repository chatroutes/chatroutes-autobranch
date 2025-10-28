"""
Intelligent routing system that automatically decides when to use LLM.

This module analyzes conversation characteristics and hybrid detection results
to intelligently decide whether LLM analysis would add value.
"""

from dataclasses import dataclass
from typing import Callable

from chatroutes_autobranch.branch_detection.conversation_analyzer import (
    ConversationFlowAnalyzer,
    ConversationTurn,
)
from chatroutes_autobranch.branch_detection.parser import LLMBranchParser


@dataclass
class RouterDecision:
    """Decision from intelligent router."""

    use_llm: bool
    confidence: float  # 0-1, how confident in the decision
    reason: str
    heuristic_scores: dict
    estimated_value: float  # Expected value-add from LLM (0-1)


class IntelligentRouter:
    """
    Automatically decides whether to use LLM based on conversation analysis.

    Uses multiple heuristics:
    1. Conversation complexity (length, ambiguity)
    2. Hybrid detection quality (branches found, confidence)
    3. Cost/benefit trade-off
    4. User constraints (time, budget)

    Example:
        >>> router = IntelligentRouter()
        >>> decision = router.should_use_llm(conversation, hybrid_results)
        >>> if decision.use_llm:
        ...     llm_branches = parser.parse(conversation)
    """

    def __init__(
        self,
        time_budget_ms: float = 10000,  # Max 10s
        cost_budget: float = 0.001,  # Max $0.001 per analysis
        min_value_threshold: float = 0.3,  # Only use LLM if expected value > 30%
        prefer_speed: bool = False,  # Optimize for speed over thoroughness
    ):
        """
        Initialize intelligent router.

        Args:
            time_budget_ms: Maximum time willing to spend on LLM (milliseconds).
            cost_budget: Maximum cost willing to spend on LLM (dollars).
            min_value_threshold: Minimum expected value-add to use LLM (0-1).
            prefer_speed: If True, bias toward hybrid-only.
        """
        self.time_budget_ms = time_budget_ms
        self.cost_budget = cost_budget
        self.min_value_threshold = min_value_threshold
        self.prefer_speed = prefer_speed

    def should_use_llm(
        self,
        conversation: list[ConversationTurn],
        hybrid_results: dict,
    ) -> RouterDecision:
        """
        Decide whether to use LLM based on heuristics.

        Args:
            conversation: List of conversation turns.
            hybrid_results: Results from hybrid analyzer.

        Returns:
            RouterDecision with recommendation and reasoning.
        """
        # Calculate heuristic scores
        scores = {}

        scores["complexity"] = self._measure_complexity(conversation)
        scores["ambiguity"] = self._measure_ambiguity(conversation)
        scores["hybrid_quality"] = self._measure_hybrid_quality(hybrid_results)
        scores["coverage"] = self._measure_coverage(conversation, hybrid_results)
        scores["implicit_signals"] = self._detect_implicit_signals(conversation)

        # Estimate value-add from LLM
        estimated_value = self._estimate_llm_value(scores)

        # Check constraints
        meets_budget = self._check_budget_constraints(conversation)
        meets_threshold = estimated_value >= self.min_value_threshold

        # Speed preference
        if self.prefer_speed and estimated_value < 0.7:
            # Only use LLM if very high expected value
            use_llm = False
            reason = "Speed preference: hybrid sufficient"
        else:
            # Normal decision logic
            use_llm = meets_budget and meets_threshold

            if use_llm:
                reason = self._generate_reason(scores, estimated_value)
            else:
                if not meets_budget:
                    reason = "Budget constraints exceeded"
                elif not meets_threshold:
                    reason = f"Expected value ({estimated_value:.2f}) below threshold ({self.min_value_threshold:.2f})"
                else:
                    reason = "Unknown"

        # Confidence in decision
        confidence = self._calculate_confidence(scores, estimated_value)

        return RouterDecision(
            use_llm=use_llm,
            confidence=confidence,
            reason=reason,
            heuristic_scores=scores,
            estimated_value=estimated_value,
        )

    def _measure_complexity(self, conversation: list[ConversationTurn]) -> float:
        """
        Measure conversation complexity (0-1).

        Factors:
        - Turn length (longer = more complex)
        - Vocabulary diversity
        - Sentence structure
        """
        if not conversation:
            return 0.0

        # Average turn length
        avg_length = sum(len(turn.content) for turn in conversation) / len(
            conversation
        )
        length_score = min(avg_length / 500, 1.0)  # Normalize to 500 chars

        # Vocabulary diversity (unique words / total words)
        all_words = []
        for turn in conversation:
            words = turn.content.lower().split()
            all_words.extend(words)

        if not all_words:
            vocab_score = 0.0
        else:
            vocab_score = len(set(all_words)) / len(all_words)

        # Combine
        complexity = (length_score * 0.6 + vocab_score * 0.4)
        return min(complexity, 1.0)

    def _measure_ambiguity(self, conversation: list[ConversationTurn]) -> float:
        """
        Measure conversation ambiguity (0-1).

        Higher ambiguity = more value from LLM.

        Signals:
        - Hedging language ("maybe", "perhaps", "might")
        - Questions without clear answers
        - Conditional statements
        - Abstract concepts
        """
        ambiguity_signals = [
            r"\b(maybe|perhaps|might|could|possibly|uncertain|unclear|ambiguous)\b",
            r"\b(depends|varies|context|situational)\b",
            r"\b(on one hand|on the other hand|however|although|but)\b",
            r"\?",  # Questions
        ]

        import re

        total_signals = 0
        total_length = 0

        for turn in conversation:
            total_length += len(turn.content)
            for pattern in ambiguity_signals:
                matches = re.findall(pattern, turn.content, re.IGNORECASE)
                total_signals += len(matches)

        if total_length == 0:
            return 0.0

        # Normalize by length (signals per 1000 chars)
        ambiguity = min((total_signals / total_length) * 1000, 1.0)
        return ambiguity

    def _measure_hybrid_quality(self, hybrid_results: dict) -> float:
        """
        Measure quality of hybrid detection (0-1).

        Lower quality = more room for LLM to add value.

        Factors:
        - Number of branches found
        - Confidence scores
        - Explicit vs semantic ratio
        """
        explicit = hybrid_results.get("explicit_branches", [])
        semantic = hybrid_results.get("semantic_branches", [])

        total_branches = len(explicit) + len(semantic)

        if total_branches == 0:
            # No branches found = low quality = high LLM value
            return 0.0

        # More branches = better quality
        branch_score = min(total_branches / 5, 1.0)  # 5+ branches = excellent

        # Explicit branches have higher quality
        if total_branches > 0:
            explicit_ratio = len(explicit) / total_branches
            quality_bonus = explicit_ratio * 0.3  # Up to +30%
        else:
            quality_bonus = 0.0

        quality = min(branch_score + quality_bonus, 1.0)
        return quality

    def _measure_coverage(
        self, conversation: list[ConversationTurn], hybrid_results: dict
    ) -> float:
        """
        Measure how well hybrid covered the conversation (0-1).

        Lower coverage = more opportunity for LLM.

        Factors:
        - Turns with detected branches
        - Turn distribution
        """
        if not conversation:
            return 1.0  # Empty = fully covered

        turns_with_branches = set()

        for branch in hybrid_results.get("explicit_branches", []):
            if "turn_id" in branch.meta:
                turns_with_branches.add(branch.meta["turn_id"])

        for branch in hybrid_results.get("semantic_branches", []):
            turns_with_branches.add(branch.turn_id)

        coverage = len(turns_with_branches) / len(conversation)
        return coverage

    def _detect_implicit_signals(
        self, conversation: list[ConversationTurn]
    ) -> float:
        """
        Detect signals of implicit decision points (0-1).

        High signals = LLM likely to find hidden branches.

        Signals:
        - "Torn between", "considering"
        - Philosophical language
        - Complex conditionals
        - Narrative shifts
        """
        import re

        implicit_patterns = [
            r"\b(torn between|considering|contemplating|weighing)\b",
            r"\b(on balance|all things considered|taking into account)\b",
            r"\b(dilemma|trade-off|pros and cons)\b",
            r"\b(if.*then.*else|depending on|in case of)\b",
        ]

        signal_count = 0
        for turn in conversation:
            for pattern in implicit_patterns:
                if re.search(pattern, turn.content, re.IGNORECASE):
                    signal_count += 1

        # Normalize by conversation length
        signal_score = min(signal_count / len(conversation), 1.0)
        return signal_score

    def _estimate_llm_value(self, scores: dict) -> float:
        """
        Estimate expected value-add from LLM (0-1).

        Combines heuristic scores with learned weights.
        """
        # Weighted combination
        weights = {
            "complexity": 0.2,  # More complex = more LLM value
            "ambiguity": 0.3,  # More ambiguous = more LLM value
            "hybrid_quality": -0.25,  # Higher quality = less LLM value
            "coverage": -0.15,  # Higher coverage = less LLM value
            "implicit_signals": 0.3,  # More implicit = more LLM value
        }

        value = 0.0
        for metric, weight in weights.items():
            if weight < 0:
                # Inverse relationship
                value += abs(weight) * (1 - scores[metric])
            else:
                # Direct relationship
                value += weight * scores[metric]

        return min(max(value, 0.0), 1.0)

    def _check_budget_constraints(self, conversation: list[ConversationTurn]) -> bool:
        """Check if LLM usage is within budget."""
        # Estimate LLM time (rough: 100ms per turn)
        estimated_time_ms = len(conversation) * 100

        # Estimate cost (rough: $0.0001 per 1000 tokens, ~1 token per 4 chars)
        total_chars = sum(len(turn.content) for turn in conversation)
        estimated_tokens = total_chars / 4
        estimated_cost = (estimated_tokens / 1000) * 0.0001

        return (
            estimated_time_ms <= self.time_budget_ms
            and estimated_cost <= self.cost_budget
        )

    def _generate_reason(self, scores: dict, estimated_value: float) -> str:
        """Generate human-readable reason for decision."""
        reasons = []

        if scores["complexity"] > 0.6:
            reasons.append("high complexity")

        if scores["ambiguity"] > 0.5:
            reasons.append("significant ambiguity")

        if scores["hybrid_quality"] < 0.4:
            reasons.append("limited hybrid detection")

        if scores["coverage"] < 0.5:
            reasons.append("low conversation coverage")

        if scores["implicit_signals"] > 0.4:
            reasons.append("implicit decision signals detected")

        if reasons:
            reason_str = ", ".join(reasons)
            return f"LLM recommended: {reason_str} (expected value: {estimated_value:.2f})"
        else:
            return f"LLM recommended (expected value: {estimated_value:.2f})"

    def _calculate_confidence(self, scores: dict, estimated_value: float) -> float:
        """Calculate confidence in routing decision (0-1)."""
        # High confidence if:
        # - Estimated value is very high (>0.8) or very low (<0.2)
        # - Scores are consistent (all high or all low)

        # Distance from threshold
        threshold_distance = abs(estimated_value - self.min_value_threshold)
        distance_confidence = min(threshold_distance * 2, 1.0)

        # Consistency of scores
        score_values = list(scores.values())
        score_variance = (
            sum((s - sum(score_values) / len(score_values)) ** 2 for s in score_values)
            / len(score_values)
        )
        consistency_confidence = 1 - min(score_variance * 2, 1.0)

        # Combine
        confidence = (distance_confidence * 0.6 + consistency_confidence * 0.4)
        return confidence


class AutoRouter:
    """
    Convenience wrapper that automatically routes and executes.

    IMPORTANT: LLM usage is OPT-IN, not automatic!

    By default, only hybrid detection is used.
    To enable LLM routing, set enable_llm_routing=True.

    Example:
        >>> # Default: Hybrid only (fast, free)
        >>> router = AutoRouter(analyzer)
        >>> results = router.analyze(conversation)

        >>> # Opt-in: Use LLM intelligently when beneficial
        >>> router = AutoRouter(analyzer, llm_parser, enable_llm_routing=True)
        >>> results = router.analyze(conversation)
    """

    def __init__(
        self,
        analyzer: ConversationFlowAnalyzer,
        llm_parser: LLMBranchParser | None = None,
        router: IntelligentRouter | None = None,
        enable_llm_routing: bool = False,
    ):
        """
        Initialize auto router.

        Args:
            analyzer: Hybrid analyzer.
            llm_parser: Optional LLM parser.
            router: Optional custom router (uses default if None).
            enable_llm_routing: If True, router may decide to use LLM.
                                If False, LLM is never used (default).
        """
        self.analyzer = analyzer
        self.llm_parser = llm_parser
        self.router = router or IntelligentRouter()
        self.enable_llm_routing = enable_llm_routing

    def analyze(
        self, conversation: list[ConversationTurn], verbose: bool = False
    ) -> dict:
        """
        Automatically analyze with intelligent LLM routing.

        Args:
            conversation: Conversation to analyze.
            verbose: If True, print decision reasoning.

        Returns:
            Dictionary with:
            - hybrid_results: Results from hybrid analyzer
            - llm_branches: LLM branches (if used)
            - router_decision: RouterDecision object (None if LLM disabled)
            - all_branches: Combined list of all branches
        """
        # Phase 1: Hybrid analysis (always)
        hybrid_results = self.analyzer.analyze(conversation)

        # Phase 2: Check if LLM routing is enabled
        if not self.enable_llm_routing:
            if verbose:
                print("\n🚀 LLM Routing: DISABLED (hybrid only mode)")
                print("   Set enable_llm_routing=True to enable intelligent LLM routing")

            # Return hybrid-only results
            return {
                "hybrid_results": hybrid_results,
                "llm_branches": [],
                "router_decision": None,
                "all_branches": (
                    hybrid_results["explicit_branches"]
                    + hybrid_results["semantic_branches"]
                ),
            }

        # Phase 3: Intelligent routing decision (only if enabled)
        decision = self.router.should_use_llm(conversation, hybrid_results)

        if verbose:
            print(f"\n🤖 Router Decision: {'USE LLM' if decision.use_llm else 'SKIP LLM'}")
            print(f"   Confidence: {decision.confidence:.2f}")
            print(f"   Reason: {decision.reason}")
            print(f"   Expected Value: {decision.estimated_value:.2f}")
            print(f"\n   Heuristic Scores:")
            for metric, score in decision.heuristic_scores.items():
                print(f"     {metric}: {score:.2f}")

        # Phase 4: LLM analysis (conditional)
        llm_branches = []
        if decision.use_llm and self.llm_parser:
            conversation_text = "\n\n".join([
                f"[{turn.speaker}]: {turn.content}" for turn in conversation
            ])
            llm_branches = self.llm_parser.parse(conversation_text)

            if verbose:
                print(f"\n   LLM detected {len(llm_branches)} additional branches")

        # Combine results
        all_branches = (
            hybrid_results["explicit_branches"]
            + hybrid_results["semantic_branches"]
            + llm_branches
        )

        return {
            "hybrid_results": hybrid_results,
            "llm_branches": llm_branches,
            "router_decision": decision,
            "all_branches": all_branches,
        }
