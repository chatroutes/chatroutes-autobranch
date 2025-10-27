"""Deterministic branch point extraction using pattern matching."""

import re
from math import prod
from typing import Any

from chatroutes_autobranch.branch_detection.models import BranchOption, BranchPoint


class BranchExtractor:
    """
    Extract branch points from text using deterministic pattern matching.

    This class identifies decision points in text where multiple mutually-exclusive
    options are presented, such as:
    - Numbered/bulleted lists (enumerations)
    - Disjunctions ("or", "either...or", "alternatively")
    - Conditionals ("if...then...else")
    - Open directives ("choose a...", "pick from...")

    Usage:
        >>> extractor = BranchExtractor()
        >>> text = '''
        ... Options:
        ... 1. Flask
        ... 2. FastAPI
        ... Database: Postgres or Mongo
        ... '''
        >>> branch_points = extractor.extract(text)
        >>> len(branch_points)
        2
        >>> extractor.count_max_leaves(branch_points)
        4
    """

    def __init__(self):
        """Initialize the branch extractor."""
        # Patterns for different branch types
        self.enumeration_pattern = re.compile(r"^(\d+\.\s+|[-*•]\s+)", re.MULTILINE)
        self.disjunction_pattern = re.compile(
            r"\b(or|either|alternatively)\b", re.IGNORECASE
        )
        self.conditional_pattern = re.compile(
            r"\b(if\b.*?\bthen\b|unless\b|otherwise\b)", re.IGNORECASE
        )
        self.directive_pattern = re.compile(
            r"\b(choose|pick|select|decide)\b.*?\b(from|between|among)\b",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> list[BranchPoint]:
        """
        Extract all branch points from text.

        Args:
            text: The text to analyze (typically an LLM response).

        Returns:
            List of BranchPoint objects found in the text.

        Examples:
            >>> extractor = BranchExtractor()
            >>> text = "Choose: 1. Flask or 2. FastAPI"
            >>> points = extractor.extract(text)
            >>> len(points) > 0
            True
        """
        branch_points = []

        # 1. Extract enumerations (bullets/numbered lists)
        enum_points = self._extract_enumerations(text)
        branch_points.extend(enum_points)

        # 2. Extract disjunctions ("or" patterns)
        disj_points = self._extract_disjunctions(text)
        branch_points.extend(disj_points)

        # 3. Extract conditionals ("if...then" patterns)
        cond_points = self._extract_conditionals(text)
        branch_points.extend(cond_points)

        return branch_points

    def _extract_enumerations(self, text: str) -> list[BranchPoint]:
        """
        Extract enumerated lists (bullets, numbered items).

        Args:
            text: Text to analyze.

        Returns:
            List of BranchPoint objects for enumerations.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        branch_points = []
        current_group = []
        group_start_idx = 0

        for idx, line in enumerate(lines):
            if self.enumeration_pattern.match(line):
                if not current_group:
                    group_start_idx = idx
                current_group.append(line)
            else:
                # End of enumeration group
                if len(current_group) >= 2:  # Must have at least 2 options
                    branch_point = self._create_enumeration_branch(
                        current_group, f"bp_enum_{group_start_idx}"
                    )
                    branch_points.append(branch_point)
                current_group = []

        # Handle last group
        if len(current_group) >= 2:
            branch_point = self._create_enumeration_branch(
                current_group, f"bp_enum_{group_start_idx}"
            )
            branch_points.append(branch_point)

        return branch_points

    def _create_enumeration_branch(
        self, lines: list[str], branch_id: str
    ) -> BranchPoint:
        """
        Create a BranchPoint from enumerated lines.

        Args:
            lines: List of enumerated lines.
            branch_id: Unique ID for this branch point.

        Returns:
            BranchPoint object.
        """
        options = []
        for idx, line in enumerate(lines):
            # Strip enumeration prefix (1., -, *, etc.)
            label = re.sub(r"^(\d+\.\s+|[-*•]\s+)", "", line).strip()
            option = BranchOption(
                id=f"{branch_id}_opt{idx}", label=label, span=line
            )
            options.append(option)

        return BranchPoint(
            id=branch_id,
            type="enumeration",
            options=options,
            context="\n".join(lines),
        )

    def _extract_disjunctions(self, text: str) -> list[BranchPoint]:
        """
        Extract disjunctions (A or B or C patterns).

        Args:
            text: Text to analyze.

        Returns:
            List of BranchPoint objects for disjunctions.
        """
        branch_points = []
        sentences = self._split_sentences(text)

        for idx, sentence in enumerate(sentences):
            if not self.disjunction_pattern.search(sentence):
                continue

            # Split on "or" to find options
            # Handle "either A or B" and "A or B or C"
            options = self._parse_disjunction_options(sentence)

            if len(options) >= 2:
                branch_id = f"bp_disj_{idx}"
                branch_options = [
                    BranchOption(id=f"{branch_id}_opt{i}", label=opt, span=opt)
                    for i, opt in enumerate(options)
                ]

                branch_point = BranchPoint(
                    id=branch_id,
                    type="disjunction",
                    options=branch_options,
                    context=sentence,
                )
                branch_points.append(branch_point)

        return branch_points

    def _parse_disjunction_options(self, sentence: str) -> list[str]:
        """
        Parse options from a disjunction sentence.

        Args:
            sentence: Sentence containing disjunction.

        Returns:
            List of option strings.
        """
        # Remove "either" prefix if present
        sentence = re.sub(r"\beither\b", "", sentence, flags=re.IGNORECASE).strip()

        # Split on "or" (case-insensitive)
        parts = re.split(r"\bor\b", sentence, flags=re.IGNORECASE)

        # Clean up each part
        options = []
        for part in parts:
            # Remove common noise words and punctuation
            cleaned = part.strip(" ,.;:!?")
            # Remove leading "alternatively", "also", etc.
            cleaned = re.sub(
                r"^\s*(alternatively|also|and)\s+", "", cleaned, flags=re.IGNORECASE
            )
            if cleaned and len(cleaned) > 1:
                options.append(cleaned)

        return options

    def _extract_conditionals(self, text: str) -> list[BranchPoint]:
        """
        Extract conditional branches (if...then...else patterns).

        Args:
            text: Text to analyze.

        Returns:
            List of BranchPoint objects for conditionals.
        """
        branch_points = []
        sentences = self._split_sentences(text)

        for idx, sentence in enumerate(sentences):
            if not self.conditional_pattern.search(sentence):
                continue

            # Simple if-then-else detection
            if_match = re.search(
                r"\bif\b\s+(.*?)\s+\bthen\b\s+(.*?)(?:\s+\belse\b\s+(.*?))?$",
                sentence,
                re.IGNORECASE,
            )

            if if_match:
                condition = if_match.group(1).strip()
                then_clause = if_match.group(2).strip()
                else_clause = if_match.group(3).strip() if if_match.group(3) else None

                branch_id = f"bp_cond_{idx}"
                options = [
                    BranchOption(
                        id=f"{branch_id}_then",
                        label=then_clause,
                        span=f"if {condition} then {then_clause}",
                    )
                ]

                if else_clause:
                    options.append(
                        BranchOption(
                            id=f"{branch_id}_else",
                            label=else_clause,
                            span=f"else {else_clause}",
                        )
                    )

                if len(options) >= 2:
                    branch_point = BranchPoint(
                        id=branch_id,
                        type="conditional",
                        options=options,
                        context=sentence,
                        meta={"condition": condition},
                    )
                    branch_points.append(branch_point)

        return branch_points

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting (can be improved with nltk/spacy)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def count_max_leaves(self, branch_points: list[BranchPoint]) -> int:
        """
        Calculate maximum number of leaf nodes (Π ki).

        For independent branch points with k1, k2, ..., kn options,
        the maximum number of leaves is the product k1 × k2 × ... × kn.

        Args:
            branch_points: List of branch points.

        Returns:
            Maximum number of possible leaf nodes.

        Examples:
            >>> extractor = BranchExtractor()
            >>> bp1 = BranchPoint(
            ...     id="bp1", type="enumeration",
            ...     options=[BranchOption(id="o1", label="A", span="A"),
            ...              BranchOption(id="o2", label="B", span="B")]
            ... )
            >>> bp2 = BranchPoint(
            ...     id="bp2", type="enumeration",
            ...     options=[BranchOption(id="o3", label="X", span="X"),
            ...              BranchOption(id="o4", label="Y", span="Y"),
            ...              BranchOption(id="o5", label="Z", span="Z")]
            ... )
            >>> extractor.count_max_leaves([bp1, bp2])
            6
        """
        if not branch_points:
            return 1

        # For now, assume all branches are independent
        # TODO: Handle dependent branches (depends_on field)
        return prod(bp.option_count for bp in branch_points)

    def count_unique_leaves(
        self, branch_points: list[BranchPoint], consider_dependencies: bool = False
    ) -> int:
        """
        Calculate number of unique leaves considering dependencies.

        Args:
            branch_points: List of branch points.
            consider_dependencies: If True, accounts for depends_on relationships.

        Returns:
            Number of unique leaf nodes.

        Note:
            Currently treats all branches as independent. Dependency
            handling will be added in future versions.
        """
        if not consider_dependencies:
            return self.count_max_leaves(branch_points)

        # TODO: Implement dependency-aware counting
        # This would build a DAG and count paths through it
        return self.count_max_leaves(branch_points)

    def get_statistics(self, branch_points: list[BranchPoint]) -> dict[str, Any]:
        """
        Get statistics about detected branch points.

        Args:
            branch_points: List of branch points.

        Returns:
            Dictionary with statistics.

        Examples:
            >>> extractor = BranchExtractor()
            >>> text = "Options: 1. A or 2. B. Choose X or Y."
            >>> points = extractor.extract(text)
            >>> stats = extractor.get_statistics(points)
            >>> stats['total_branch_points'] >= 0
            True
        """
        if not branch_points:
            return {
                "total_branch_points": 0,
                "total_options": 0,
                "max_leaves": 1,
                "by_type": {},
                "avg_options_per_branch": 0.0,
            }

        by_type = {}
        total_options = 0

        for bp in branch_points:
            by_type[bp.type] = by_type.get(bp.type, 0) + 1
            total_options += bp.option_count

        return {
            "total_branch_points": len(branch_points),
            "total_options": total_options,
            "max_leaves": self.count_max_leaves(branch_points),
            "by_type": by_type,
            "avg_options_per_branch": total_options / len(branch_points),
        }
