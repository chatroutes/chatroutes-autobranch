"""LLM-based branch point parsing for complex cases."""

import json
from typing import Any, Callable

from chatroutes_autobranch.branch_detection.models import BranchOption, BranchPoint


class LLMBranchParser:
    """
    Use an LLM to parse branch points when deterministic patterns fail.

    This class uses an LLM as a structured parser (not a generator) to identify
    branch points in complex or ambiguous text. It enforces a strict JSON schema
    to ensure the LLM only extracts existing options, not invents new ones.

    Usage:
        >>> def my_llm(prompt: str) -> str:
        ...     # Your LLM call here
        ...     return '{"branch_points": []}'
        >>>
        >>> parser = LLMBranchParser(llm=my_llm)
        >>> text = "You could use either approach, depending on context."
        >>> branch_points = parser.parse(text)

    Note:
        This is an optional component. Most cases should work with
        the deterministic BranchExtractor. Use LLMBranchParser only when:
        - Text is ambiguous or unstructured
        - Domain knowledge is required to identify options
        - Pattern matching fails to capture implicit choices
    """

    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        """
        Initialize the LLM branch parser.

        Args:
            llm: Callable that takes a prompt string and returns LLM response.
                 Should accept temperature parameter.
            temperature: LLM temperature (lower = more deterministic).
            max_retries: Maximum number of retry attempts on parse errors.
        """
        self.llm = llm
        self.temperature = temperature
        self.max_retries = max_retries

        # System prompt for the LLM
        self.system_prompt = """You are a precise text analyzer that extracts branch points from text.

A branch point is a span in the text where ≥2 mutually-exclusive options are explicitly or implicitly presented.

Types of branch points:
- Enumeration: Numbered or bulleted lists (e.g., "1. Flask 2. FastAPI")
- Disjunction: "or" patterns (e.g., "use Flask or FastAPI")
- Conditional: if-then-else patterns (e.g., "if you need speed, use FastAPI; else use Flask")
- Open directive: choice points (e.g., "choose a framework that fits your needs")

Return strict JSON:
{
  "branch_points": [
    {
      "id": "bp1",
      "type": "enumeration|disjunction|conditional|open_directive",
      "options": [
        {"id": "opt1", "label": "Flask", "span": "1. Flask"},
        {"id": "opt2", "label": "FastAPI", "span": "2. FastAPI"}
      ],
      "context": "original text span",
      "depends_on": []
    }
  ]
}

CRITICAL RULES:
1. Only extract EXPLICIT options from the text - do not invent new ones
2. Each branch point must have ≥2 options
3. Options must be mutually exclusive
4. Use exact text spans from the original text
5. Return valid JSON only (no markdown, no explanation)
"""

    def parse(
        self, text: str, fallback_to_empty: bool = True
    ) -> list[BranchPoint]:
        """
        Parse branch points from text using LLM.

        Args:
            text: Text to analyze.
            fallback_to_empty: If True, return empty list on errors; if False, raise.

        Returns:
            List of BranchPoint objects.

        Raises:
            ValueError: If LLM not configured or response invalid (when fallback_to_empty=False).
        """
        if self.llm is None:
            if fallback_to_empty:
                return []
            raise ValueError("LLM not configured. Provide llm parameter to constructor.")

        # Construct prompt
        user_prompt = f"""Extract branch points from this text:

TEXT:
{text}

Return JSON only (no markdown, no explanation):"""

        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"

        # Try parsing with retries
        for attempt in range(self.max_retries + 1):
            try:
                # Call LLM
                response = self.llm(full_prompt)

                # Parse JSON response
                branch_points = self._parse_response(response)
                return branch_points

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < self.max_retries:
                    # Retry with slightly different prompt
                    continue
                else:
                    if fallback_to_empty:
                        return []
                    raise ValueError(f"Failed to parse LLM response after {self.max_retries} retries: {e}")

        return []

    def _parse_response(self, response: str) -> list[BranchPoint]:
        """
        Parse LLM JSON response into BranchPoint objects.

        Args:
            response: LLM response string (should be JSON).

        Returns:
            List of BranchPoint objects.

        Raises:
            json.JSONDecodeError: If response is not valid JSON.
            KeyError: If required fields are missing.
            ValueError: If data validation fails.
        """
        # Strip markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            # Remove ```json and ``` markers
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        # Parse JSON
        data = json.loads(response)

        if "branch_points" not in data:
            raise KeyError("Response missing 'branch_points' field")

        branch_points = []
        for bp_data in data["branch_points"]:
            # Parse options
            options = []
            for opt_data in bp_data["options"]:
                option = BranchOption(
                    id=opt_data["id"],
                    label=opt_data["label"],
                    span=opt_data.get("span", opt_data["label"]),
                    meta=opt_data.get("meta", {}),
                )
                options.append(option)

            # Create branch point
            branch_point = BranchPoint(
                id=bp_data["id"],
                type=bp_data["type"],
                options=options,
                depends_on=bp_data.get("depends_on", []),
                context=bp_data.get("context", ""),
                meta=bp_data.get("meta", {}),
            )
            branch_points.append(branch_point)

        return branch_points

    def parse_with_confidence(
        self, text: str
    ) -> tuple[list[BranchPoint], dict[str, Any]]:
        """
        Parse branch points and return confidence metrics.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (branch_points, metadata) where metadata includes:
            - success: bool
            - error: str | None
            - attempts: int

        Examples:
            >>> parser = LLMBranchParser(llm=lambda p: '{"branch_points": []}')
            >>> points, meta = parser.parse_with_confidence("Some text")
            >>> meta["success"]
            True
        """
        metadata = {"success": False, "error": None, "attempts": 0}

        try:
            branch_points = self.parse(text, fallback_to_empty=False)
            metadata["success"] = True
            metadata["attempts"] = 1
            return branch_points, metadata

        except Exception as e:
            metadata["error"] = str(e)
            metadata["attempts"] = self.max_retries + 1
            return [], metadata
