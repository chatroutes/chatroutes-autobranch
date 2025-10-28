# Hybrid Branch Detection Guide

## Overview

The `ConversationFlowAnalyzer` combines two powerful approaches:

1. **Explicit Branch Detection** (Deterministic) - You mark branches manually
2. **Semantic Branch Detection** (AI-powered) - Automatically detects implicit branches

## Quick Start

```python
from chatroutes_autobranch.branch_detection import (
    ConversationFlowAnalyzer,
    ConversationTurn,
)
from chatroutes_autobranch.core.embeddings import DummyEmbeddingProvider

# Create conversation
conversation = [
    ConversationTurn(id="1", speaker="user", content="What about X?"),
    ConversationTurn(id="2", speaker="assistant", content="..."),
]

# Analyze with hybrid approach
analyzer = ConversationFlowAnalyzer(
    embedding_provider=DummyEmbeddingProvider(dimension=384, seed=42),
    topic_shift_threshold=0.6,
    enable_explicit=True,
    enable_semantic=True,
)

results = analyzer.analyze(conversation)
```

## Explicit Branch Detection

### How to Mark Branches

Add explicit markers in your conversation:

```
BRANCH: Choose an approach
OPTIONS:
1. Option A - Description
2. Option B - Description
3. Option C - Description
```

Supported markers:
- `BRANCH:`
- `OPTIONS:`
- `CHOOSE:`
- `DECISION:`

### Features

- **100% confidence** - Deterministic detection
- **Clear options** - Numbered or bulleted lists
- **Actionable** - Direct mapping to decisions
- **Flexible** - Works with any list format (1., 2., -, *, etc.)

### Example Result

```python
results['explicit_branches']
# [BranchPoint(
#     id="bp_enum_0",
#     type="enumeration",
#     options=[
#         BranchOption(label="Option A - Description"),
#         BranchOption(label="Option B - Description"),
#     ],
#     meta={'confidence': 1.0, 'turn_id': '2'}
# )]
```

## Semantic Branch Detection

### Automatic Detection Types

1. **Topic Shifts** - Cosine similarity < threshold between consecutive turns
2. **Decision Points** - "I'm wondering..." → "I want to..."
3. **Question-to-Action** - "What should I do?" → "I'll create..."

### Configuration

```python
analyzer = ConversationFlowAnalyzer(
    embedding_provider=embedding_provider,
    topic_shift_threshold=0.6,  # Lower = more sensitive
    enable_semantic=True,
)
```

### Detected Patterns

**Topic Shift:**
```
Turn 1: "Let's discuss LLM training data problems"
Turn 2: "I want to write about human value preservation"
→ DETECTED: Low cosine similarity (0.4 < 0.6)
```

**Decision Point:**
```
Turn 1: "I'm wondering if I should focus on philosophy..."
Turn 2: "I want to write a practical guide"
→ DETECTED: Curiosity → Commitment
```

**Question-to-Action:**
```
Turn 1: "What should I write about?"
Turn 2: "I'll create a multi-article series"
→ DETECTED: Question → Action verb
```

### Example Result

```python
results['semantic_branches']
# [SemanticBranch(
#     turn_id="5",
#     branch_type="decision_point",
#     description="Transition from exploration to commitment",
#     confidence=0.80,
#     context_before="I'm wondering about...",
#     context_after="I want to write..."
# )]
```

## Combined Graph

The analyzer merges both detection methods:

```python
graph = results['combined_graph']

# Graph structure
{
    'nodes': [
        {'id': '1', 'type': 'turn', 'content': '...'},
        {'id': 'explicit_0', 'type': 'explicit_branch', 'confidence': 1.0},
        {'id': 'semantic_0', 'type': 'semantic_branch', 'confidence': 0.8},
    ],
    'edges': [
        {'from': '2', 'to': 'explicit_0'},
        {'from': '5', 'to': 'semantic_0'},
    ]
}
```

## Statistics

```python
stats = results['statistics']

{
    'total_branches': 3,
    'explicit_count': 1,
    'semantic_count': 2,
    'explicit_options': 3,
    'semantic_types': {
        'topic_shift': 1,
        'decision_point': 1,
    },
    'avg_semantic_confidence': 0.75,
}
```

## When to Use Each Approach

### Use Explicit Markers When:

- ✅ You know the decision points ahead of time
- ✅ You want 100% confidence/reliability
- ✅ You need clear, actionable options
- ✅ You're designing structured workflows

**Example Use Cases:**
- LLM-generated multiple choice questions
- Workflow decision trees
- Interactive tutorials
- Structured problem-solving

### Use Semantic Detection When:

- ✅ Analyzing natural conversation
- ✅ Discovering hidden branch points
- ✅ Understanding conversation flow
- ✅ Post-hoc analysis of chat logs

**Example Use Cases:**
- Conversation analysis
- Chat log mining
- User journey mapping
- Research on decision patterns

### Use Both (Hybrid) When:

- ✅ You want comprehensive coverage
- ✅ You mark major branches explicitly
- ✅ You want to discover subtle shifts
- ✅ Building intelligent conversation systems

**Example Use Cases:**
- Conversational AI with branching
- Complex decision support systems
- Research + production hybrid
- Your article research workflow!

## Real-World Example: Your Article Project

### Without Explicit Markers

```python
# Just analyze natural conversation
results = analyzer.analyze(your_conversation)

# Found 1 semantic branch:
# - Decision point at turn 8 (confidence: 0.80)
#   "here is my rough draft" → "i want to add historical references"
```

### With Explicit Markers

```python
# Add marker at decision point
ConversationTurn(
    id="7a",
    speaker="user",
    content="""
BRANCH: Article Approach
OPTIONS:
1. Philosophical - Focus on meaning, human value
2. Practical - Focus on professions, actionable advice
3. Historical - Focus on labor transitions, patterns
""",
)

results = analyzer.analyze(enhanced_conversation)

# Found 3 total branches:
# - 1 explicit (3 options, confidence: 1.0)
# - 2 semantic (avg confidence: 0.80)
```

## Advanced: Custom Semantic Detection

You can extend semantic detection with custom patterns:

```python
class MyCustomAnalyzer(ConversationFlowAnalyzer):
    def _detect_scope_change(self, turn1, turn2):
        """Detect when scope narrows or broadens."""
        # Your custom logic here
        pass

    def _detect_semantic_branches(self, conversation):
        branches = super()._detect_semantic_branches(conversation)

        # Add your custom detections
        for i in range(len(conversation) - 1):
            scope_branch = self._detect_scope_change(
                conversation[i],
                conversation[i + 1]
            )
            if scope_branch:
                branches.append(scope_branch)

        return branches
```

## Integration with LLMBranchParser

For even more powerful semantic detection:

```python
from chatroutes_autobranch.branch_detection import LLMBranchParser

# Use LLM for complex semantic parsing
llm_parser = LLMBranchParser(llm=your_llm_function)

# Combine all three approaches
explicit = analyzer._detect_explicit_branches(conversation)
semantic = analyzer._detect_semantic_branches(conversation)
llm_detected = llm_parser.parse(conversation_text)

# Merge all results
all_branches = explicit + semantic + llm_detected
```

## Tips & Best Practices

1. **Start with explicit markers** for critical decision points
2. **Use semantic detection** for exploration and discovery
3. **Tune threshold** (0.5 = sensitive, 0.7 = conservative)
4. **Inspect confidence scores** - trust high confidence branches
5. **Validate semantic branches** - they're heuristic-based
6. **Combine with LLM parser** for ambiguous cases

## Example Output Analysis

From your conversation analysis:

```
SEMANTIC DETECTION FOUND:
- 1 decision point (confidence: 0.80)
  Turn 7 → 8: "rough draft" → "i want to add references"

EXPLICIT DETECTION FOUND (after adding marker):
- 1 enumeration (confidence: 1.0)
  3 options: Philosophical, Practical, Historical

COMBINED:
- 3 total branches
- 3 explicit options to explore
- Clear decision point identified
- Unified graph with 18 nodes, 3 edges
```

## Next Steps

1. Try it on your conversations
2. Add explicit markers at key decision points
3. Analyze the semantic branches discovered
4. Build a conversation flow map
5. Use insights to structure your article series!

## Full Example Script

See `examples/analyze_conversation_hybrid.py` for a complete working example.

---

**Built with chatroutes-autobranch v1.2.0+**
