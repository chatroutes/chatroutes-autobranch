# Quick Start Guide

This guide will help you get started with developing and using chatroutes-autobranch.

## Installation

### For Users

Install from PyPI (when published):

```bash
pip install chatroutes-autobranch
```

### For Developers

1. Clone the repository (or you're already in it):

```bash
cd C:\Users\afzal\chatroutes\chatroutes-autobranch
```

2. Install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

## Running Tests

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=chatroutes_autobranch --cov-report=html
```

Run specific test file:

```bash
pytest tests/test_candidate.py
```

## Running Examples

Basic usage:

```bash
python examples/basic_usage.py
```

Config-based usage:

```bash
python examples/config_based.py
```

## Code Quality

Format code:

```bash
black src tests examples
```

Lint code:

```bash
ruff check src tests examples
```

Type check:

```bash
mypy src
```

Or run all pre-commit hooks manually:

```bash
pre-commit run --all-files
```

## Project Structure

```
chatroutes-autobranch/
├── src/chatroutes_autobranch/    # Main package
│   ├── core/                      # Core dataclasses and protocols
│   │   ├── candidate.py           # Candidate, ScoredCandidate
│   │   ├── protocols.py           # Protocol interfaces
│   │   ├── selector.py            # BranchSelector (main pipeline)
│   │   ├── scorer.py              # Scorer implementations
│   │   └── embeddings.py          # Embedding providers
│   ├── beam/                      # Beam search
│   │   └── selector.py            # BeamSelector
│   ├── novelty/                   # Novelty filtering
│   │   ├── cosine.py              # Cosine similarity filter
│   │   └── mmr.py                 # MMR filter
│   ├── entropy/                   # Entropy stopping
│   │   └── shannon.py             # Shannon entropy stopper
│   ├── budget/                    # Budget management
│   │   └── manager.py             # Budget, BudgetManager
│   └── config/                    # Configuration
│       └── loader.py              # Config loader
├── tests/                         # Test suite
├── examples/                      # Usage examples
├── .github/workflows/             # CI/CD
├── README.md                      # Main documentation
├── CONTRIBUTING.md               # Contribution guide
├── LICENSE                       # Apache 2.0
└── pyproject.toml                # Package configuration
```

## Development Workflow

1. **Make changes** to the code
2. **Write tests** for your changes
3. **Run tests** to verify: `pytest`
4. **Format and lint**: `pre-commit run --all-files`
5. **Commit** your changes
6. **Push** and create a pull request

## Key TODOs

The following components have skeleton implementations with TODOs:

1. **CompositeScorer** (src/chatroutes_autobranch/core/scorer.py)
   - Implement weighted composite scoring
   - See chatroutes_autobranch_v1.0.md for algorithm details

2. **CosineNoveltyFilter** (src/chatroutes_autobranch/novelty/cosine.py)
   - Implement cosine similarity filtering
   - Compute embeddings and filter duplicates

3. **MMRNoveltyFilter** (src/chatroutes_autobranch/novelty/mmr.py)
   - Implement Maximal Marginal Relevance algorithm
   - Balance relevance and diversity

4. **ShannonEntropyStopper** (src/chatroutes_autobranch/entropy/shannon.py)
   - Implement K-means clustering
   - Compute Shannon entropy on cluster distributions

5. **BranchSelector.from_config()** (src/chatroutes_autobranch/core/selector.py)
   - Parse config and instantiate components
   - Factory method for config-based initialization

## Next Steps

1. Review the specification: `chatroutes_autobranch_v1.0.md`
2. Run the examples: `python examples/basic_usage.py`
3. Run the tests: `pytest`
4. Pick a TODO and start implementing!
5. See CONTRIBUTING.md for detailed contribution guidelines

## Questions?

- Read the spec: `chatroutes_autobranch_v1.0.md`
- Check examples: `examples/`
- Review tests: `tests/`
- See contributing guide: `CONTRIBUTING.md`

Happy coding!
