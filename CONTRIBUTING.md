# Contributing to chatroutes-autobranch

Thank you for your interest in contributing to chatroutes-autobranch! We welcome contributions from the community and are excited to work with you.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to hello@chatroutes.com.

**In short:**
- Be respectful and inclusive
- Welcome newcomers
- Focus on what's best for the community
- Show empathy towards others

---

## Getting Started

### Ways to Contribute

We welcome many types of contributions:

- 🐛 **Bug reports** - Found a bug? Open an issue with reproducible steps
- 💡 **Feature requests** - Have an idea? Start a discussion
- 📝 **Documentation** - Improve README, docs, docstrings, examples
- 🧪 **Tests** - Add test coverage, property tests, benchmarks
- 🔧 **Code** - Bug fixes, new features, performance improvements
- 🎨 **Examples** - Real-world usage examples, integrations
- 💬 **Community** - Answer questions, help others in issues/discussions

### Good First Issues

Look for issues labeled `good-first-issue` – these are beginner-friendly and well-scoped. We're happy to mentor first-time contributors!

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, or poetry)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/chatroutes-autobranch.git
   cd chatroutes-autobranch
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/chatroutes/chatroutes-autobranch.git
   ```

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode with dev dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest tests/

# Check types
mypy src/

# Check formatting
black --check src/ tests/
ruff check src/ tests/
```

If everything passes, you're ready to contribute! 🎉

---

## Making Changes

### Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or: bugfix/issue-number-description
# or: docs/what-you-are-documenting
```

**Branch naming conventions:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation improvements
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements
- `perf/` - Performance improvements

### Make Your Changes

1. **Write tests first** (TDD approach recommended)
2. Implement your changes
3. Ensure tests pass
4. Update documentation if needed
5. Add entry to CHANGELOG.md under `[Unreleased]`

### Commit Messages

Use clear, descriptive commit messages:

**Good examples:**
```
Add MMR novelty filter implementation

Implement Maximal Marginal Relevance algorithm for novelty filtering.
Includes lambda parameter for diversity/relevance tradeoff and
comprehensive tests.

Fixes #42
```

```
Fix entropy calculation for single-candidate edge case

Shannon entropy was returning NaN when only one candidate remained.
Now correctly returns 0.0 and sets should_continue=False.

Closes #73
```

**Format:**
- First line: Short summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)
- Reference issues/PRs if applicable

**Prefixes:**
- `Add:` - New feature
- `Fix:` - Bug fix
- `Refactor:` - Code restructuring
- `Docs:` - Documentation only
- `Test:` - Test additions
- `Perf:` - Performance improvement

---

## Code Standards

### Python Style

We follow PEP 8 with these tools:

- **Black** (formatting, line length 100)
- **Ruff** (linting)
- **MyPy** (type checking)

```bash
# Auto-format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/

# Type checking
mypy src/
```

### Type Hints

All public functions must have type hints:

```python
# Good
def score(self, parent: Candidate, candidates: list[Candidate]) -> list[ScoredCandidate]:
    """Score candidates relative to parent."""
    ...

# Bad (no types)
def score(self, parent, candidates):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def prune(self, candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """
    Prune similar candidates from a score-descending list.

    Args:
        candidates: List of scored candidates, sorted by score descending.
                   BeamSelector guarantees this ordering.

    Returns:
        Subset of candidates with similar items removed, maintaining
        score order.

    Raises:
        ValueError: If candidates list is not sorted by score.

    Example:
        >>> filter = CosineNoveltyFilter(threshold=0.85)
        >>> kept = filter.prune(scored_candidates)
    """
```

### Code Organization

- One class per file (exceptions for small helper classes)
- Group related functions together
- Use `__all__` to define public API
- Keep functions focused (< 50 lines if possible)

---

## Testing

### Writing Tests

All new features must have tests:

```python
# tests/test_your_feature.py
import pytest
from chatroutes_autobranch import YourFeature

def test_basic_functionality():
    """Test basic use case."""
    feature = YourFeature(param=value)
    result = feature.method(input)
    assert result == expected

def test_edge_case_empty_input():
    """Test edge case: empty input."""
    feature = YourFeature()
    result = feature.method([])
    assert result == []

@pytest.mark.parametrize("threshold,expected", [
    (0.5, 3),
    (0.8, 2),
    (0.95, 1),
])
def test_threshold_variations(threshold, expected):
    """Test different threshold values."""
    ...
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_beam.py

# Run with coverage
pytest tests/ --cov=chatroutes_autobranch --cov-report=html

# Run specific test
pytest tests/test_beam.py::test_select_topk

# Run fast tests only (skip slow benchmarks)
pytest tests/ -m "not slow"
```

### Test Requirements

- **Unit tests** for all new functions/classes
- **Edge cases** (empty input, single item, large input)
- **Error cases** (invalid input, exceptions)
- **Integration tests** for component interactions
- **Property tests** using Hypothesis (when applicable)

**Coverage target:** 90%+ for new code

---

## Submitting Changes

### Before Submitting

Checklist:

- [ ] Tests pass: `pytest tests/`
- [ ] Type checking passes: `mypy src/`
- [ ] Linting passes: `ruff check src/ tests/`
- [ ] Formatting correct: `black --check src/ tests/`
- [ ] Documentation updated (if API changed)
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Commit messages are clear

### Create Pull Request

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create Pull Request

3. Fill out PR template:
   - **Description**: What does this PR do?
   - **Motivation**: Why is this change needed?
   - **Testing**: How was this tested?
   - **Checklist**: Complete the checklist

4. Link related issues: `Fixes #123` or `Closes #456`

### PR Review Process

1. **Automated checks** run (tests, linting, type checking)
2. **Maintainer review** (usually within 2-3 business days)
3. **Address feedback** by pushing new commits
4. **Approval** from at least one maintainer required
5. **Merge** (squash merge for clean history)

### After Merge

- Your contribution will be included in the next release
- You'll be added to CONTRIBUTORS.md
- Thank you! 🎉

---

## Release Process

*(For maintainers)*

### Version Bumping

1. Update version in `pyproject.toml`
2. Move `[Unreleased]` items to new version in CHANGELOG.md
3. Create git tag: `git tag v0.x.0`
4. Push tag: `git push origin v0.x.0`

### Publishing to PyPI

```bash
# Build
python -m build

# Test on TestPyPI first
python -m twine upload --repository testpypi dist/*

# Verify installation
pip install --index-url https://test.pypi.org/simple/ chatroutes-autobranch

# Publish to PyPI
python -m twine upload dist/*
```

### GitHub Release

1. Go to Releases → Draft new release
2. Choose tag (v0.x.0)
3. Generate release notes (auto from PRs)
4. Highlight major changes
5. Publish release

---

## Getting Help

### Communication Channels

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - Questions, ideas, general discussion
- **Email** - hello@chatroutes.com (for sensitive issues)

### Questions?

Before opening an issue, check:

1. [README.md](./README.md) - Quick start and examples
2. [Technical Specification](./chatroutes_autobranch_v1.0.md) - Complete API docs
3. [Existing Issues](https://github.com/chatroutes/chatroutes-autobranch/issues) - Maybe already answered
4. [Discussions](https://github.com/chatroutes/chatroutes-autobranch/discussions) - Community Q&A

Still stuck? Open a discussion or issue – we're happy to help!

---

## Recognition

We value all contributions! Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Thanked in our community updates

---

## Areas We'd Love Help With

**High Priority:**
- Additional novelty algorithms (DPP, k-DPP)
- More embedding providers (Cohere, Voyage AI, local models)
- Adaptive K scheduling (auto-tune beam width)
- Performance optimizations (FAISS integration, caching)

**Nice to Have:**
- Tree visualization tools
- More integration examples (specific frameworks)
- Jupyter notebook tutorials
- Benchmark comparisons with other tools

**Documentation:**
- Video tutorials
- Blog posts about use cases
- Translation to other languages

---

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to chatroutes-autobranch!** 🚀

Questions? Reach out in [Discussions](https://github.com/chatroutes/chatroutes-autobranch/discussions) or email hello@chatroutes.com
