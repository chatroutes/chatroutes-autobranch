"""Pytest configuration and shared fixtures."""

import pytest
from chatroutes_autobranch.core.candidate import Candidate, ScoredCandidate
from chatroutes_autobranch.core.embeddings import DummyEmbeddingProvider
from chatroutes_autobranch.core.scorer import StaticScorer


@pytest.fixture
def sample_parent() -> Candidate:
    """Sample parent candidate for testing."""
    return Candidate(
        id="parent-1",
        text="What is the capital of France?",
        meta={"source": "test"},
    )


@pytest.fixture
def sample_candidates() -> list[Candidate]:
    """Sample candidate list for testing."""
    return [
        Candidate(id="c1", text="Paris is the capital of France."),
        Candidate(id="c2", text="The capital is Paris."),
        Candidate(id="c3", text="Lyon is a city in France."),
        Candidate(id="c4", text="France's capital is Paris."),
        Candidate(id="c5", text="Berlin is the capital of Germany."),
    ]


@pytest.fixture
def sample_scored_candidates() -> list[ScoredCandidate]:
    """Sample scored candidates for testing."""
    return [
        ScoredCandidate(id="c1", text="Paris", score=0.9),
        ScoredCandidate(id="c2", text="Lyon", score=0.7),
        ScoredCandidate(id="c3", text="Marseille", score=0.6),
        ScoredCandidate(id="c4", text="Nice", score=0.5),
        ScoredCandidate(id="c5", text="Toulouse", score=0.4),
    ]


@pytest.fixture
def dummy_embedding_provider() -> DummyEmbeddingProvider:
    """Dummy embedding provider for testing."""
    return DummyEmbeddingProvider(dimension=128, seed=42)


@pytest.fixture
def static_scorer() -> StaticScorer:
    """Static scorer for testing."""
    return StaticScorer({})
