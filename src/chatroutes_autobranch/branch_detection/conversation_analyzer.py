"""
Hybrid conversation flow analyzer combining explicit and semantic branch detection.

This module provides tools to analyze conversations and identify both:
1. Explicit branches (manually marked by user)
2. Implicit branches (detected via semantic analysis)
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vectors."""
        ...


from chatroutes_autobranch.branch_detection.extractor import BranchExtractor
from chatroutes_autobranch.branch_detection.models import BranchOption, BranchPoint


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    id: str
    speaker: str  # "user" or "assistant"
    content: str
    timestamp: int | None = None  # Optional ordering
    meta: dict = field(default_factory=dict)


@dataclass
class SemanticBranch:
    """A semantically detected branch point in conversation."""

    turn_id: str
    branch_type: str  # "topic_shift", "decision_point", "question_to_action"
    description: str
    confidence: float  # 0.0 to 1.0
    context_before: str
    context_after: str
    meta: dict = field(default_factory=dict)


class ConversationFlowAnalyzer:
    """
    Hybrid analyzer for conversation branches.

    Combines:
    1. Explicit branch detection (user-marked branches)
    2. Semantic branch detection (AI-powered topic/intent shifts)

    Usage:
        >>> analyzer = ConversationFlowAnalyzer()
        >>> conversation = [
        ...     ConversationTurn(id="1", speaker="user", content="What about X?"),
        ...     ConversationTurn(id="2", speaker="assistant", content="BRANCH: Choose A or B\\n1. A\\n2. B"),
        ... ]
        >>> results = analyzer.analyze(conversation)
        >>> results['explicit_branches']  # Deterministic
        >>> results['semantic_branches']  # AI-detected
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        topic_shift_threshold: float = 0.6,
        enable_explicit: bool = True,
        enable_semantic: bool = True,
    ):
        """
        Initialize the conversation flow analyzer.

        Args:
            embedding_provider: Provider for semantic similarity (optional).
            topic_shift_threshold: Similarity threshold for topic shifts (0-1).
                                   Lower = more sensitive to topic changes.
            enable_explicit: Enable explicit branch detection.
            enable_semantic: Enable semantic branch detection.
        """
        self.extractor = BranchExtractor()
        self.embedding_provider = embedding_provider
        self.topic_shift_threshold = topic_shift_threshold
        self.enable_explicit = enable_explicit
        self.enable_semantic = enable_semantic

        # Patterns for explicit branch markers
        self.explicit_markers = [
            r"^BRANCH:",
            r"^OPTIONS?:",
            r"^CHOOSE:",
            r"^DECISION:",
        ]

    def analyze(
        self, conversation: list[ConversationTurn]
    ) -> dict:
        """
        Analyze conversation for all branch types.

        Args:
            conversation: List of conversation turns.

        Returns:
            Dictionary with:
            - explicit_branches: List of explicitly marked BranchPoint objects
            - semantic_branches: List of SemanticBranch objects
            - combined_graph: Unified representation
            - statistics: Analysis stats

        Examples:
            >>> analyzer = ConversationFlowAnalyzer()
            >>> turns = [
            ...     ConversationTurn(id="1", speaker="user", content="Help me choose"),
            ...     ConversationTurn(id="2", speaker="assistant", content="BRANCH: A or B\\n1. A\\n2. B"),
            ... ]
            >>> results = analyzer.analyze(turns)
            >>> len(results['explicit_branches']) > 0
            True
        """
        results = {
            "explicit_branches": [],
            "semantic_branches": [],
            "combined_graph": None,
            "statistics": {},
        }

        # 1. Detect explicit branches
        if self.enable_explicit:
            explicit = self._detect_explicit_branches(conversation)
            results["explicit_branches"] = explicit

        # 2. Detect semantic branches
        if self.enable_semantic:
            semantic = self._detect_semantic_branches(conversation)
            results["semantic_branches"] = semantic

        # 3. Combine into unified graph
        results["combined_graph"] = self._build_combined_graph(
            conversation,
            results["explicit_branches"],
            results["semantic_branches"],
        )

        # 4. Calculate statistics
        results["statistics"] = self._calculate_statistics(results)

        return results

    def _detect_explicit_branches(
        self, conversation: list[ConversationTurn]
    ) -> list[BranchPoint]:
        """
        Detect explicitly marked branches using markers like "BRANCH:", "OPTIONS:", etc.

        Args:
            conversation: List of conversation turns.

        Returns:
            List of BranchPoint objects found via explicit markers.
        """
        explicit_branches = []

        for turn in conversation:
            # Check if turn contains explicit branch marker
            has_marker = any(
                re.search(pattern, turn.content, re.IGNORECASE | re.MULTILINE)
                for pattern in self.explicit_markers
            )

            if has_marker:
                # Extract branch points from this turn
                branches = self.extractor.extract(turn.content)

                # Annotate with turn metadata
                for branch in branches:
                    branch.meta["turn_id"] = turn.id
                    branch.meta["speaker"] = turn.speaker
                    branch.meta["detection_type"] = "explicit"
                    branch.meta["confidence"] = 1.0

                explicit_branches.extend(branches)

        return explicit_branches

    def _detect_semantic_branches(
        self, conversation: list[ConversationTurn]
    ) -> list[SemanticBranch]:
        """
        Detect implicit semantic branches via topic shifts, decision points, etc.

        Args:
            conversation: List of conversation turns.

        Returns:
            List of SemanticBranch objects.
        """
        semantic_branches = []

        if not self.embedding_provider:
            # Cannot do semantic analysis without embeddings
            return semantic_branches

        # Analyze pairs of consecutive turns
        for i in range(len(conversation) - 1):
            current = conversation[i]
            next_turn = conversation[i + 1]

            # Detect topic shifts
            topic_shift = self._detect_topic_shift(current, next_turn)
            if topic_shift:
                semantic_branches.append(topic_shift)

            # Detect decision points
            decision = self._detect_decision_point(current, next_turn)
            if decision:
                semantic_branches.append(decision)

            # Detect question-to-action transitions
            q2a = self._detect_question_to_action(current, next_turn)
            if q2a:
                semantic_branches.append(q2a)

        return semantic_branches

    def _detect_topic_shift(
        self, turn1: ConversationTurn, turn2: ConversationTurn
    ) -> SemanticBranch | None:
        """
        Detect if there's a significant topic shift between two turns.

        Args:
            turn1: First turn.
            turn2: Second turn.

        Returns:
            SemanticBranch if shift detected, None otherwise.
        """
        if not self.embedding_provider:
            return None

        # Get embeddings for both turns
        emb1 = self.embedding_provider.embed([turn1.content])[0]
        emb2 = self.embedding_provider.embed([turn2.content])[0]

        # Calculate cosine similarity
        similarity = self._cosine_similarity(emb1, emb2)

        # If similarity below threshold, it's a topic shift
        if similarity < self.topic_shift_threshold:
            return SemanticBranch(
                turn_id=turn2.id,
                branch_type="topic_shift",
                description=f"Topic shift detected (similarity: {similarity:.2f})",
                confidence=1.0 - similarity,  # Lower similarity = higher confidence
                context_before=turn1.content[:200],
                context_after=turn2.content[:200],
                meta={
                    "similarity": similarity,
                    "threshold": self.topic_shift_threshold,
                },
            )

        return None

    def _detect_decision_point(
        self, turn1: ConversationTurn, turn2: ConversationTurn
    ) -> SemanticBranch | None:
        """
        Detect decision points (curiosity -> commitment transitions).

        Patterns:
        - "I'm wondering..." -> "I want to..."
        - "What about..." -> "Let's focus on..."
        - Questions -> Declarative statements

        Args:
            turn1: First turn.
            turn2: Second turn.

        Returns:
            SemanticBranch if decision detected, None otherwise.
        """
        # Pattern matching for decision signals
        curiosity_patterns = [
            r"\b(wondering|curious|what if|should i|could i)\b",
            r"\?",  # Questions
        ]

        commitment_patterns = [
            r"\b(i want|i will|let's|i'd like|i need)\b",
            r"\b(decided|choose|focus on|go with)\b",
        ]

        has_curiosity = any(
            re.search(p, turn1.content, re.IGNORECASE) for p in curiosity_patterns
        )

        has_commitment = any(
            re.search(p, turn2.content, re.IGNORECASE) for p in commitment_patterns
        )

        if has_curiosity and has_commitment:
            return SemanticBranch(
                turn_id=turn2.id,
                branch_type="decision_point",
                description="Transition from exploration to commitment",
                confidence=0.8,  # Heuristic-based, moderate confidence
                context_before=turn1.content[:200],
                context_after=turn2.content[:200],
                meta={
                    "curiosity_signals": has_curiosity,
                    "commitment_signals": has_commitment,
                },
            )

        return None

    def _detect_question_to_action(
        self, turn1: ConversationTurn, turn2: ConversationTurn
    ) -> SemanticBranch | None:
        """
        Detect question-to-action transitions.

        Args:
            turn1: First turn (should contain question).
            turn2: Second turn (should contain action).

        Returns:
            SemanticBranch if transition detected, None otherwise.
        """
        # Check if turn1 ends with question mark
        has_question = turn1.content.strip().endswith("?")

        # Check if turn2 starts with action verbs
        action_verbs = [
            r"^(i'll|let's|i will|i'm going to|i would like to)",
            r"^(create|build|write|implement|design|analyze)",
        ]

        has_action = any(
            re.search(p, turn2.content, re.IGNORECASE | re.MULTILINE)
            for p in action_verbs
        )

        if has_question and has_action:
            return SemanticBranch(
                turn_id=turn2.id,
                branch_type="question_to_action",
                description="Transition from question to actionable plan",
                confidence=0.7,
                context_before=turn1.content[:200],
                context_after=turn2.content[:200],
                meta={"question": has_question, "action": has_action},
            )

        return None

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Similarity score (0 to 1).
        """
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _build_combined_graph(
        self,
        conversation: list[ConversationTurn],
        explicit: list[BranchPoint],
        semantic: list[SemanticBranch],
    ) -> dict:
        """
        Build unified graph combining explicit and semantic branches.

        Args:
            conversation: Original conversation turns.
            explicit: Explicit branch points.
            semantic: Semantic branch points.

        Returns:
            Dictionary representing combined branch graph.
        """
        # Simple graph structure: list of nodes and edges
        nodes = []
        edges = []

        # Add conversation turns as nodes
        for turn in conversation:
            nodes.append(
                {
                    "id": turn.id,
                    "type": "turn",
                    "speaker": turn.speaker,
                    "content": turn.content[:100] + "..."
                    if len(turn.content) > 100
                    else turn.content,
                }
            )

        # Add explicit branches as nodes
        for i, branch in enumerate(explicit):
            branch_id = f"explicit_{i}"
            nodes.append(
                {
                    "id": branch_id,
                    "type": "explicit_branch",
                    "branch_type": branch.type,
                    "options": [opt.label for opt in branch.options],
                    "confidence": 1.0,
                }
            )

            # Link to turn where it appears
            if "turn_id" in branch.meta:
                edges.append({"from": branch.meta["turn_id"], "to": branch_id})

        # Add semantic branches as nodes
        for i, branch in enumerate(semantic):
            branch_id = f"semantic_{i}"
            nodes.append(
                {
                    "id": branch_id,
                    "type": "semantic_branch",
                    "branch_type": branch.branch_type,
                    "description": branch.description,
                    "confidence": branch.confidence,
                }
            )

            # Link to turn where it occurs
            edges.append({"from": branch.turn_id, "to": branch_id})

        return {"nodes": nodes, "edges": edges}

    def _calculate_statistics(self, results: dict) -> dict:
        """
        Calculate statistics about detected branches.

        Args:
            results: Analysis results dictionary.

        Returns:
            Statistics dictionary.
        """
        explicit = results["explicit_branches"]
        semantic = results["semantic_branches"]

        stats = {
            "total_branches": len(explicit) + len(semantic),
            "explicit_count": len(explicit),
            "semantic_count": len(semantic),
            "explicit_options": sum(bp.option_count for bp in explicit),
            "semantic_types": {},
        }

        # Count semantic branch types
        for branch in semantic:
            btype = branch.branch_type
            stats["semantic_types"][btype] = stats["semantic_types"].get(btype, 0) + 1

        # Average confidence for semantic branches
        if semantic:
            stats["avg_semantic_confidence"] = sum(
                b.confidence for b in semantic
            ) / len(semantic)
        else:
            stats["avg_semantic_confidence"] = 0.0

        return stats
