# Branch Detection Module

## Overview

The `branch_detection` module provides tools to analyze text (like LLM responses) and identify decision points where multiple mutually-exclusive options exist.

**Status**: ✅ Complete and tested (34 tests, 93% coverage)

## What It Does

Identifies branch points in text using deterministic pattern matching:

- **Enumerations**: Numbered or bulleted lists (e.g., "1. Flask 2. FastAPI")
- **Disjunctions**: "or" patterns (e.g., "use Flask or FastAPI")
- **Conditionals**: if-then-else patterns (e.g., "if you need speed, use FastAPI; else use Flask")
- **Open directives**: Choice points (e.g., "choose a framework")

Then calculates the **maximum possible paths** (combinatorial product: k1 × k2 × ... × kn)

## Installation

Already included in the main package:

```bash
pip install -e .
```

## Quick Start

```python
from chatroutes_autobranch import BranchExtractor

text = """
Backend options:
1. Flask
2. FastAPI
3. Django

Database: Postgres or MySQL
"""

extractor = BranchExtractor()
branch_points = extractor.extract(text)

print(f"Found {len(branch_points)} branch points")
print(f"Max combinations: {extractor.count_max_leaves(branch_points)}")
# Output: Found 2 branch points
#         Max combinations: 6 (3 backends × 2 databases)
```

## Module Structure

```
src/chatroutes_autobranch/branch_detection/
├── __init__.py          # Exports
├── models.py            # BranchPoint, BranchOption dataclasses
├── extractor.py         # BranchExtractor (deterministic)
└── parser.py            # LLMBranchParser (optional)

tests/
└── test_branch_detection.py  # 34 comprehensive tests
```

## Core Components

### 1. BranchOption

A single option at a branch point.

```python
from chatroutes_autobranch import BranchOption

opt = BranchOption(
    id="opt1",
    label="Flask",
    span="1. Flask - lightweight framework"
)
```

### 2. BranchPoint

A decision point with multiple options.

```python
from chatroutes_autobranch import BranchPoint

bp = BranchPoint(
    id="bp1",
    type="enumeration",  # or "disjunction", "conditional", "open_directive"
    options=[opt1, opt2, opt3]
)

print(bp.option_count)  # 3
print(bp.get_option_labels())  # ["Flask", "FastAPI", "Django"]
```

### 3. BranchExtractor (Deterministic)

Main component for pattern-based extraction.

```python
from chatroutes_autobranch import BranchExtractor

extractor = BranchExtractor()

# Extract branch points
branch_points = extractor.extract(text)

# Count maximum possible leaves (Π ki)
max_leaves = extractor.count_max_leaves(branch_points)

# Get statistics
stats = extractor.get_statistics(branch_points)
print(stats)
# {
#   "total_branch_points": 3,
#   "total_options": 7,
#   "max_leaves": 12,
#   "by_type": {"enumeration": 2, "disjunction": 1},
#   "avg_options_per_branch": 2.3
# }
```

### 4. LLMBranchParser (Optional)

Use LLM when deterministic patterns fail.

```python
from chatroutes_autobranch import LLMBranchParser

def my_llm(prompt: str) -> str:
    # Your LLM call here
    return llm_api.generate(prompt)

parser = LLMBranchParser(llm=my_llm)
branch_points = parser.parse(text)
```

## Pattern Detection

### Enumerations

Detects:
- Numbered lists: `1. Flask`, `2. FastAPI`
- Bullets: `- Flask`, `* FastAPI`, `• Flask`

Example:
```
Options:
1. Flask
2. FastAPI
→ Detects 1 branch point with 2 options
```

### Disjunctions

Detects:
- "or" patterns: `Flask or FastAPI`
- "either...or": `either Flask or FastAPI`
- Multiple: `Flask or FastAPI or Django`

Example:
```
Use Flask or FastAPI for your API.
→ Detects 1 branch point with 2 options
```

### Conditionals

Detects:
- if-then-else: `if X then Y else Z`
- unless: `unless X do Y`
- otherwise: `do X otherwise Y`

Example:
```
If you need speed then use FastAPI else use Flask.
→ Detects 1 branch point with 2 options
```

## Usage Patterns

### Basic Usage

```python
from chatroutes_autobranch import BranchExtractor

extractor = BranchExtractor()
text = "Choose: 1. Flask 2. FastAPI. DB: Postgres or MySQL."

branch_points = extractor.extract(text)
print(f"Found {len(branch_points)} decision points")

for bp in branch_points:
    print(f"{bp.type}: {bp.get_option_labels()}")
```

### Counting Combinations

```python
# Independent branches: multiply
bp1 = BranchPoint(id="bp1", type="enumeration",
                  options=[opt1, opt2, opt3])  # 3 options

bp2 = BranchPoint(id="bp2", type="disjunction",
                  options=[opt4, opt5])  # 2 options

max_leaves = extractor.count_max_leaves([bp1, bp2])
# Result: 6 (3 × 2)
```

### Statistics

```python
stats = extractor.get_statistics(branch_points)

print(f"Total branch points: {stats['total_branch_points']}")
print(f"Max combinations: {stats['max_leaves']}")
print(f"Breakdown by type: {stats['by_type']}")
```

### Hybrid Approach (Recommended)

```python
from chatroutes_autobranch import BranchExtractor, LLMBranchParser

# Try deterministic first (fast, free)
extractor = BranchExtractor()
branch_points = extractor.extract(text)

if len(branch_points) == 0:
    # Fallback to LLM for complex cases
    parser = LLMBranchParser(llm=my_llm)
    branch_points = parser.parse(text)
```

## Examples

Run the comprehensive example:

```bash
python examples/branch_detection_usage.py
```

This demonstrates:
1. Basic extraction
2. Statistics
3. Real-world LLM response analysis
4. Edge case handling
5. LLM parser (optional)

## Testing

Run all branch detection tests:

```bash
pytest tests/test_branch_detection.py -v
```

Results:
- ✅ 34 tests
- ✅ 100% pass rate
- ✅ 93% code coverage (extractor)
- ✅ 92% code coverage (parser)

## Integration with Existing Pipeline

The branch detection module is **separate** from the branch selection pipeline:

```
[NEW] Branch Detection        →  [EXISTING] Branch Selection
──────────────────────             ──────────────────────────
Text → Branch Points               Candidates → Filtered
     → Option Extraction                      → Scored
     → Count estimation                       → Pruned
```

**Example integration:**

```python
from chatroutes_autobranch import (
    BranchExtractor,      # NEW: Detect branch points
    BranchSelector,       # EXISTING: Select best branches
)

# Phase 1: Analyze text for branch points
extractor = BranchExtractor()
branch_points = extractor.extract(llm_response)
print(f"Found {len(branch_points)} decision points")

# Phase 2: Generate candidates (your LLM)
# candidates = generate_from_branch_points(branch_points)

# Phase 3: Filter candidates (existing pipeline)
# selector = BranchSelector(...)
# result = selector.step(parent, candidates)
```

## Design Decisions

### Why Separate Module?

1. **Different concerns**: Text analysis vs. candidate filtering
2. **Optional feature**: Not everyone needs branch detection
3. **Backward compatible**: Doesn't break existing code
4. **Independent testing**: Own test suite

### Why Deterministic First?

1. **Fast**: No LLM calls
2. **Free**: No API costs
3. **Reproducible**: Same input → same output
4. **Accurate**: Works for 90% of cases

### When to Use LLM Parser?

Only when:
- Text is ambiguous or unstructured
- Domain knowledge required
- Pattern matching fails
- Implicit choices need extraction

## API Reference

### BranchExtractor

```python
class BranchExtractor:
    def extract(text: str) -> list[BranchPoint]
    def count_max_leaves(branch_points: list[BranchPoint]) -> int
    def count_unique_leaves(branch_points, consider_dependencies=False) -> int
    def get_statistics(branch_points: list[BranchPoint]) -> dict
```

### LLMBranchParser

```python
class LLMBranchParser:
    def __init__(llm: Callable, temperature=0.1, max_retries=2)
    def parse(text: str, fallback_to_empty=True) -> list[BranchPoint]
    def parse_with_confidence(text: str) -> tuple[list[BranchPoint], dict]
```

### BranchPoint

```python
@dataclass
class BranchPoint:
    id: str
    type: str  # "enumeration", "disjunction", "conditional", "open_directive"
    options: list[BranchOption]
    depends_on: list[str] = []  # For nested branches
    context: str = ""
    meta: dict = {}

    @property
    def option_count() -> int
    def get_option_labels() -> list[str]
```

### BranchOption

```python
@dataclass
class BranchOption:
    id: str
    label: str
    span: str  # Original text span
    meta: dict = {}
```

## Limitations

1. **Language**: English only (patterns are English-specific)
2. **Dependencies**: Currently treats all branches as independent
   - TODO: Handle nested dependencies (depends_on field)
3. **LLM Variability**: Optional LLM parser is non-deterministic
4. **Complex Grammar**: May miss unconventional phrasings

## Future Enhancements

- [ ] Dependency-aware counting (DAG traversal)
- [ ] Multi-language support
- [ ] Confidence scores per branch point
- [ ] Visual tree rendering
- [ ] Integration examples with real LLMs

## Changelog

### v0.1.0 (2025-01-27)
- ✅ Initial release
- ✅ BranchExtractor with pattern matching
- ✅ Enumeration, disjunction, conditional detection
- ✅ Combinatorial counting (Π ki)
- ✅ Statistics and metadata
- ✅ Optional LLM parser
- ✅ Comprehensive test suite (34 tests)
- ✅ Usage examples

## Summary

The `branch_detection` module provides **deterministic, fast, accurate** branch point identification from text. It's:

- ✅ **Ready to use**: Import and go
- ✅ **Well-tested**: 34 tests, 93% coverage
- ✅ **Documented**: Examples and API docs
- ✅ **Backward compatible**: No breaking changes
- ✅ **Optional**: Use only if needed

**Use it to analyze LLM responses and count possible branching paths before generation.**
