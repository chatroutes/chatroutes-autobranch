"""
chatroutes-autobranch: Controlled branching generation for LLM applications.

This library provides clean, standalone primitives for:
- Beam search with composite scoring
- Novelty pruning (cosine similarity, MMR)
- Entropy-based stopping
- Budget management
- Pluggable, testable, replaceable components
"""

__version__ = "0.1.0"

# Core dataclasses
from chatroutes_autobranch.core.candidate import (
    Candidate,
    ScoredCandidate,
)

# Protocols
from chatroutes_autobranch.core.protocols import (
    EmbeddingProvider,
    Scorer,
    NoveltyFilter,
    EntropyStopper,
)

# Beam search
from chatroutes_autobranch.beam.selector import BeamSelector

# Novelty filtering
from chatroutes_autobranch.novelty.cosine import CosineNoveltyFilter
from chatroutes_autobranch.novelty.mmr import MMRNoveltyFilter

# Entropy stopping
from chatroutes_autobranch.entropy.shannon import ShannonEntropyStopper

# Budget management
from chatroutes_autobranch.budget.manager import (
    Budget,
    BudgetManager,
    BudgetExceededError,
)

# Branch selector (main pipeline)
from chatroutes_autobranch.core.selector import (
    BranchSelector,
    SelectionResult,
)

# Embedding providers
from chatroutes_autobranch.core.embeddings import (
    DummyEmbeddingProvider,
)

# Scorers
from chatroutes_autobranch.core.scorer import (
    CompositeScorer,
    StaticScorer,
)

# Config
from chatroutes_autobranch.config.loader import (
    load_config,
    create_default_config,
)

__all__ = [
    # Version
    "__version__",
    # Core dataclasses
    "Candidate",
    "ScoredCandidate",
    # Protocols
    "EmbeddingProvider",
    "Scorer",
    "NoveltyFilter",
    "EntropyStopper",
    # Beam
    "BeamSelector",
    # Novelty
    "CosineNoveltyFilter",
    "MMRNoveltyFilter",
    # Entropy
    "ShannonEntropyStopper",
    # Budget
    "Budget",
    "BudgetManager",
    "BudgetExceededError",
    # Main pipeline
    "BranchSelector",
    "SelectionResult",
    # Embeddings
    "DummyEmbeddingProvider",
    # Scorers
    "CompositeScorer",
    "StaticScorer",
    # Config
    "load_config",
    "create_default_config",
]
