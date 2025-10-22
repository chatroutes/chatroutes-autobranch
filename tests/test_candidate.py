"""Tests for Candidate and ScoredCandidate dataclasses."""

import math
import warnings
import pytest
from chatroutes_autobranch.core.candidate import Candidate, ScoredCandidate


class TestCandidate:
    """Tests for Candidate dataclass."""

    def test_create_basic_candidate(self):
        """Test creating a basic candidate."""
        c = Candidate(id="c1", text="Paris")
        assert c.id == "c1"
        assert c.text == "Paris"
        assert c.meta == {}

    def test_create_candidate_with_meta(self):
        """Test creating a candidate with metadata."""
        c = Candidate(id="c1", text="Paris", meta={"source": "llm", "score": 0.9})
        assert c.meta["source"] == "llm"
        assert c.meta["score"] == 0.9

    def test_candidate_id_required(self):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="Candidate id cannot be empty"):
            Candidate(id="", text="Paris")

    def test_candidate_equality(self):
        """Test candidate equality comparison."""
        c1 = Candidate(id="c1", text="Paris")
        c2 = Candidate(id="c1", text="Paris")
        c3 = Candidate(id="c2", text="Paris")

        assert c1 == c2
        assert c1 != c3


class TestScoredCandidate:
    """Tests for ScoredCandidate dataclass."""

    def test_create_scored_candidate(self):
        """Test creating a scored candidate."""
        sc = ScoredCandidate(id="c1", text="Paris", score=0.9)
        assert sc.id == "c1"
        assert sc.text == "Paris"
        assert sc.score == 0.9

    def test_score_defaults_to_zero(self):
        """Test that score defaults to 0.0."""
        sc = ScoredCandidate(id="c1", text="Paris")
        assert sc.score == 0.0

    def test_score_clamped_to_zero_one(self):
        """Test that scores are clamped to [0, 1]."""
        # Score > 1
        sc1 = ScoredCandidate(id="c1", text="Paris", score=1.5)
        assert sc1.score == 1.0

        # Score < 0
        sc2 = ScoredCandidate(id="c2", text="Lyon", score=-0.5)
        assert sc2.score == 0.0

        # Valid score
        sc3 = ScoredCandidate(id="c3", text="Nice", score=0.7)
        assert sc3.score == 0.7

    def test_nan_score_handled(self):
        """Test that NaN scores are converted to 0.0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sc = ScoredCandidate(id="c1", text="Paris", score=math.nan)
            assert sc.score == 0.0
            assert len(w) == 1
            assert "Invalid score" in str(w[0].message)

    def test_inf_score_handled(self):
        """Test that infinite scores are converted to 0.0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sc = ScoredCandidate(id="c1", text="Paris", score=math.inf)
            assert sc.score == 0.0
            assert len(w) == 1
            assert "Invalid score" in str(w[0].message)
