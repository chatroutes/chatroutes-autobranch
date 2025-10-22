"""Tests for BeamSelector."""

import pytest
from chatroutes_autobranch.beam.selector import BeamSelector
from chatroutes_autobranch.core.candidate import Candidate, ScoredCandidate
from chatroutes_autobranch.core.scorer import StaticScorer


class TestBeamSelector:
    """Tests for BeamSelector."""

    def test_beam_selector_initialization(self, static_scorer):
        """Test BeamSelector initialization."""
        selector = BeamSelector(scorer=static_scorer, k=3)
        assert selector.k == 3
        assert selector.scorer == static_scorer

    def test_k_validation(self, static_scorer):
        """Test that k must be >= 1."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            BeamSelector(scorer=static_scorer, k=0)

        with pytest.raises(ValueError, match="k must be >= 1"):
            BeamSelector(scorer=static_scorer, k=-1)

    def test_select_top_k(self, sample_parent, sample_candidates, static_scorer):
        """Test selecting top-K candidates."""
        selector = BeamSelector(scorer=static_scorer, k=3)
        result = selector.select(sample_parent, sample_candidates)

        assert len(result) <= 3
        # All should be ScoredCandidate
        assert all(isinstance(c, ScoredCandidate) for c in result)

    def test_select_fewer_than_k(self, sample_parent, static_scorer):
        """Test selecting when candidates < k."""
        selector = BeamSelector(scorer=static_scorer, k=5)
        candidates = [
            Candidate(id="c1", text="Paris"),
            Candidate(id="c2", text="Lyon"),
        ]
        result = selector.select(sample_parent, candidates)

        assert len(result) == 2

    def test_select_empty_candidates(self, sample_parent, static_scorer):
        """Test selecting from empty candidate list."""
        selector = BeamSelector(scorer=static_scorer, k=3)
        result = selector.select(sample_parent, [])

        assert result == []

    def test_deterministic_tie_breaking(self, sample_parent):
        """Test deterministic tie-breaking using lexicographic ID ordering."""
        # Create scorer that gives same score to all (empty dict = default 0.5)
        scorer = StaticScorer({})
        selector = BeamSelector(scorer=scorer, k=2)

        # Candidates with same scores but different IDs
        candidates = [
            Candidate(id="c3", text="Third"),
            Candidate(id="c1", text="First"),
            Candidate(id="c2", text="Second"),
        ]

        result = selector.select(sample_parent, candidates)

        # Should select c1 and c2 (lexicographically first after tie-break)
        assert len(result) == 2
        assert result[0].id == "c1"
        assert result[1].id == "c2"

    def test_score_ordering(self, sample_parent):
        """Test that results are ordered by score descending."""
        # Create a custom scorer that returns different scores
        class CustomScorer:
            def score(self, parent, candidates):
                scores = {"c1": 0.9, "c2": 0.7, "c3": 0.5}
                return [
                    ScoredCandidate(id=c.id, text=c.text, score=scores.get(c.id, 0.0))
                    for c in candidates
                ]

        selector = BeamSelector(scorer=CustomScorer(), k=3)
        candidates = [
            Candidate(id="c3", text="Low"),
            Candidate(id="c1", text="High"),
            Candidate(id="c2", text="Mid"),
        ]

        result = selector.select(sample_parent, candidates)

        # Should be ordered: c1 (0.9), c2 (0.7), c3 (0.5)
        assert result[0].id == "c1"
        assert result[1].id == "c2"
        assert result[2].id == "c3"
