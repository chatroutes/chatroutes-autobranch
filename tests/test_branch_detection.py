"""Tests for branch detection module."""

import json
import pytest

from chatroutes_autobranch.branch_detection import (
    BranchExtractor,
    BranchOption,
    BranchPoint,
    LLMBranchParser,
)


class TestBranchOption:
    """Tests for BranchOption model."""

    def test_create_basic_option(self):
        """Test creating a basic branch option."""
        opt = BranchOption(id="opt1", label="Flask", span="1. Flask")
        assert opt.id == "opt1"
        assert opt.label == "Flask"
        assert opt.span == "1. Flask"
        assert opt.meta == {}

    def test_create_option_with_meta(self):
        """Test creating option with metadata."""
        opt = BranchOption(
            id="opt1", label="Flask", span="1. Flask", meta={"priority": "high"}
        )
        assert opt.meta["priority"] == "high"

    def test_empty_id_raises_error(self):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            BranchOption(id="", label="Flask", span="Flask")

    def test_empty_label_raises_error(self):
        """Test that empty label raises ValueError."""
        with pytest.raises(ValueError, match="label cannot be empty"):
            BranchOption(id="opt1", label="", span="Flask")


class TestBranchPoint:
    """Tests for BranchPoint model."""

    def test_create_basic_branch_point(self):
        """Test creating a basic branch point."""
        opt1 = BranchOption(id="o1", label="Flask", span="Flask")
        opt2 = BranchOption(id="o2", label="FastAPI", span="FastAPI")

        bp = BranchPoint(id="bp1", type="enumeration", options=[opt1, opt2])

        assert bp.id == "bp1"
        assert bp.type == "enumeration"
        assert len(bp.options) == 2
        assert bp.option_count == 2
        assert bp.get_option_labels() == ["Flask", "FastAPI"]

    def test_branch_point_with_dependencies(self):
        """Test branch point with dependencies."""
        opt1 = BranchOption(id="o1", label="Flask", span="Flask")
        opt2 = BranchOption(id="o2", label="FastAPI", span="FastAPI")

        bp = BranchPoint(
            id="bp2",
            type="conditional",
            options=[opt1, opt2],
            depends_on=["bp1"],
        )

        assert bp.depends_on == ["bp1"]

    def test_empty_id_raises_error(self):
        """Test that empty ID raises ValueError."""
        opt1 = BranchOption(id="o1", label="A", span="A")
        opt2 = BranchOption(id="o2", label="B", span="B")

        with pytest.raises(ValueError, match="id cannot be empty"):
            BranchPoint(id="", type="enumeration", options=[opt1, opt2])

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises ValueError."""
        opt1 = BranchOption(id="o1", label="A", span="A")
        opt2 = BranchOption(id="o2", label="B", span="B")

        with pytest.raises(ValueError, match="type must be one of"):
            BranchPoint(id="bp1", type="invalid_type", options=[opt1, opt2])

    def test_too_few_options_raises_error(self):
        """Test that < 2 options raises ValueError."""
        opt1 = BranchOption(id="o1", label="A", span="A")

        with pytest.raises(ValueError, match="must have at least 2 options"):
            BranchPoint(id="bp1", type="enumeration", options=[opt1])

    def test_invalid_option_type_raises_error(self):
        """Test that non-BranchOption in options raises TypeError."""
        with pytest.raises(TypeError, match="must be BranchOption instances"):
            BranchPoint(id="bp1", type="enumeration", options=["not", "valid"])


class TestBranchExtractor:
    """Tests for BranchExtractor."""

    def test_extract_numbered_list(self):
        """Test extracting numbered list."""
        text = """
        Options:
        1. Flask
        2. FastAPI
        3. Django
        """

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        assert len(branch_points) == 1
        assert branch_points[0].type == "enumeration"
        assert branch_points[0].option_count == 3
        assert "Flask" in branch_points[0].get_option_labels()
        assert "FastAPI" in branch_points[0].get_option_labels()
        assert "Django" in branch_points[0].get_option_labels()

    def test_extract_bulleted_list(self):
        """Test extracting bulleted list."""
        text = """
        Choose from:
        - Postgres
        - MySQL
        - MongoDB
        """

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        assert len(branch_points) == 1
        assert branch_points[0].type == "enumeration"
        assert branch_points[0].option_count == 3

    def test_extract_disjunction(self):
        """Test extracting 'or' patterns."""
        text = "You can use Flask or FastAPI for your API."

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        assert len(branch_points) >= 1
        disj = [bp for bp in branch_points if bp.type == "disjunction"]
        assert len(disj) >= 1
        assert disj[0].option_count >= 2

    def test_extract_multiple_disjunctions(self):
        """Test extracting multiple disjunction patterns."""
        text = "Use Flask or FastAPI. Database: Postgres or Mongo or MySQL."

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        disj = [bp for bp in branch_points if bp.type == "disjunction"]
        assert len(disj) >= 2

    def test_extract_conditional(self):
        """Test extracting if-then-else pattern."""
        text = "If you need speed then use FastAPI else use Flask."

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        cond = [bp for bp in branch_points if bp.type == "conditional"]
        assert len(cond) >= 1
        assert cond[0].option_count == 2

    def test_extract_mixed_patterns(self):
        """Test extracting multiple pattern types."""
        text = """
        Backend options:
        1. Flask
        2. FastAPI

        For database, use Postgres or Mongo.

        If you need caching then use Redis else use in-memory.
        """

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        assert len(branch_points) >= 3
        types = [bp.type for bp in branch_points]
        assert "enumeration" in types
        assert "disjunction" in types
        assert "conditional" in types

    def test_count_max_leaves_single_branch(self):
        """Test counting max leaves with single branch point."""
        extractor = BranchExtractor()

        opt1 = BranchOption(id="o1", label="A", span="A")
        opt2 = BranchOption(id="o2", label="B", span="B")
        opt3 = BranchOption(id="o3", label="C", span="C")

        bp = BranchPoint(id="bp1", type="enumeration", options=[opt1, opt2, opt3])

        assert extractor.count_max_leaves([bp]) == 3

    def test_count_max_leaves_multiple_branches(self):
        """Test counting max leaves with multiple branch points."""
        extractor = BranchExtractor()

        # Branch 1: 2 options
        bp1 = BranchPoint(
            id="bp1",
            type="enumeration",
            options=[
                BranchOption(id="o1", label="A", span="A"),
                BranchOption(id="o2", label="B", span="B"),
            ],
        )

        # Branch 2: 3 options
        bp2 = BranchPoint(
            id="bp2",
            type="enumeration",
            options=[
                BranchOption(id="o3", label="X", span="X"),
                BranchOption(id="o4", label="Y", span="Y"),
                BranchOption(id="o5", label="Z", span="Z"),
            ],
        )

        # 2 × 3 = 6 leaves
        assert extractor.count_max_leaves([bp1, bp2]) == 6

    def test_count_max_leaves_empty(self):
        """Test counting max leaves with no branch points."""
        extractor = BranchExtractor()
        assert extractor.count_max_leaves([]) == 1

    def test_get_statistics(self):
        """Test getting statistics about branch points."""
        text = """
        Backend:
        1. Flask
        2. FastAPI

        Database: Postgres or Mongo or MySQL.
        """

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)
        stats = extractor.get_statistics(branch_points)

        assert stats["total_branch_points"] >= 2
        assert stats["total_options"] >= 5
        assert stats["max_leaves"] >= 6
        assert "enumeration" in stats["by_type"]
        assert "disjunction" in stats["by_type"]
        assert stats["avg_options_per_branch"] > 0

    def test_get_statistics_empty(self):
        """Test getting statistics with no branch points."""
        extractor = BranchExtractor()
        stats = extractor.get_statistics([])

        assert stats["total_branch_points"] == 0
        assert stats["total_options"] == 0
        assert stats["max_leaves"] == 1
        assert stats["by_type"] == {}
        assert stats["avg_options_per_branch"] == 0.0

    def test_ignore_single_item_list(self):
        """Test that single-item lists are not treated as branch points."""
        text = """
        Option:
        1. Only one choice
        """

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        # Should not find any branch points (need >= 2 options)
        assert len(branch_points) == 0

    def test_complex_real_world_example(self):
        """Test with complex real-world text."""
        text = """
        For your backend, you have several options:
        1. Flask - lightweight and flexible
        2. FastAPI - modern and fast
        3. Django - batteries included

        The database choice depends on your needs. You can use Postgres or MySQL
        for relational data, or MongoDB for document storage.

        If you need caching then use Redis else you can skip it.

        Hosting options include Vercel, Fly.io, or AWS.
        """

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        # Should find: enumeration (3 backends), disjunction (databases),
        # conditional (caching), disjunction (hosting)
        assert len(branch_points) >= 3

        stats = extractor.get_statistics(branch_points)
        assert stats["total_branch_points"] >= 3
        assert stats["max_leaves"] >= 12  # At least 3×2×2


class TestLLMBranchParser:
    """Tests for LLMBranchParser."""

    def test_parse_without_llm_returns_empty(self):
        """Test that parsing without LLM returns empty list."""
        parser = LLMBranchParser(llm=None)
        result = parser.parse("Some text", fallback_to_empty=True)
        assert result == []

    def test_parse_without_llm_raises_when_no_fallback(self):
        """Test that parsing without LLM raises when fallback disabled."""
        parser = LLMBranchParser(llm=None)
        with pytest.raises(ValueError, match="LLM not configured"):
            parser.parse("Some text", fallback_to_empty=False)

    def test_parse_with_valid_json_response(self):
        """Test parsing with valid JSON response from LLM."""

        def mock_llm(prompt: str) -> str:
            return json.dumps(
                {
                    "branch_points": [
                        {
                            "id": "bp1",
                            "type": "enumeration",
                            "options": [
                                {"id": "o1", "label": "Flask", "span": "Flask"},
                                {"id": "o2", "label": "FastAPI", "span": "FastAPI"},
                            ],
                            "context": "Backend options",
                        }
                    ]
                }
            )

        parser = LLMBranchParser(llm=mock_llm)
        branch_points = parser.parse("Choose Flask or FastAPI")

        assert len(branch_points) == 1
        assert branch_points[0].type == "enumeration"
        assert branch_points[0].option_count == 2

    def test_parse_with_markdown_wrapped_json(self):
        """Test parsing when LLM returns markdown-wrapped JSON."""

        def mock_llm(prompt: str) -> str:
            return """```json
{
  "branch_points": [
    {
      "id": "bp1",
      "type": "disjunction",
      "options": [
        {"id": "o1", "label": "A", "span": "A"},
        {"id": "o2", "label": "B", "span": "B"}
      ]
    }
  ]
}
```"""

        parser = LLMBranchParser(llm=mock_llm)
        branch_points = parser.parse("Choose A or B")

        assert len(branch_points) == 1

    def test_parse_with_invalid_json_returns_empty(self):
        """Test that invalid JSON returns empty list with fallback."""

        def mock_llm(prompt: str) -> str:
            return "This is not valid JSON"

        parser = LLMBranchParser(llm=mock_llm, max_retries=0)
        result = parser.parse("Some text", fallback_to_empty=True)

        assert result == []

    def test_parse_with_invalid_json_raises_without_fallback(self):
        """Test that invalid JSON raises without fallback."""

        def mock_llm(prompt: str) -> str:
            return "This is not valid JSON"

        parser = LLMBranchParser(llm=mock_llm, max_retries=0)

        with pytest.raises(ValueError, match="Failed to parse"):
            parser.parse("Some text", fallback_to_empty=False)

    def test_parse_with_confidence(self):
        """Test parse_with_confidence method."""

        def mock_llm(prompt: str) -> str:
            return json.dumps({"branch_points": []})

        parser = LLMBranchParser(llm=mock_llm)
        branch_points, metadata = parser.parse_with_confidence("Some text")

        assert isinstance(branch_points, list)
        assert metadata["success"] is True
        assert metadata["error"] is None
        assert metadata["attempts"] == 1

    def test_parse_with_confidence_on_error(self):
        """Test parse_with_confidence returns error metadata."""

        def mock_llm(prompt: str) -> str:
            return "invalid json"

        parser = LLMBranchParser(llm=mock_llm, max_retries=0)
        branch_points, metadata = parser.parse_with_confidence("Some text")

        assert branch_points == []
        assert metadata["success"] is False
        assert metadata["error"] is not None

    def test_parse_multiple_branch_points(self):
        """Test parsing response with multiple branch points."""

        def mock_llm(prompt: str) -> str:
            return json.dumps(
                {
                    "branch_points": [
                        {
                            "id": "bp1",
                            "type": "enumeration",
                            "options": [
                                {"id": "o1", "label": "A", "span": "A"},
                                {"id": "o2", "label": "B", "span": "B"},
                            ],
                        },
                        {
                            "id": "bp2",
                            "type": "disjunction",
                            "options": [
                                {"id": "o3", "label": "X", "span": "X"},
                                {"id": "o4", "label": "Y", "span": "Y"},
                            ],
                        },
                    ]
                }
            )

        parser = LLMBranchParser(llm=mock_llm)
        branch_points = parser.parse("Some text")

        assert len(branch_points) == 2
        assert branch_points[0].id == "bp1"
        assert branch_points[1].id == "bp2"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_extractor_and_statistics_pipeline(self):
        """Test full pipeline: extract → count → statistics."""
        text = """
        Backend options:
        1. Flask
        2. FastAPI
        3. Django

        Database: Postgres or MySQL.
        """

        extractor = BranchExtractor()
        branch_points = extractor.extract(text)

        assert len(branch_points) >= 2

        max_leaves = extractor.count_max_leaves(branch_points)
        assert max_leaves >= 6  # 3 backends × 2 databases

        stats = extractor.get_statistics(branch_points)
        assert stats["max_leaves"] == max_leaves
        assert stats["total_branch_points"] == len(branch_points)

    def test_empty_text_handling(self):
        """Test that empty text is handled gracefully."""
        extractor = BranchExtractor()

        # Empty string
        assert extractor.extract("") == []

        # Whitespace only
        assert extractor.extract("   \n\n  ") == []

        # No branch points
        result = extractor.extract("This is just regular text with no options.")
        assert len(result) == 0
