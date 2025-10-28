# Intelligent LLM Routing

## Default Behavior: LLM is Opt-In

**By default, LLM is DISABLED.** The system only uses fast, free hybrid detection unless you explicitly enable LLM routing.

## Quick Start

```python
from chatroutes_autobranch.branch_detection import (
    AutoRouter,
    ConversationFlowAnalyzer,
    LLMBranchParser,
)

analyzer = ConversationFlowAnalyzer(...)

# DEFAULT: Hybrid only (fast, free, no LLM)
router = AutoRouter(analyzer)
results = router.analyze(conversation)
# ✓ Uses only hybrid detection
# ✓ Instant results
# ✓ $0 cost

# OPT-IN: Enable intelligent LLM routing
llm_parser = LLMBranchParser(llm=your_llm_function)
router = AutoRouter(
    analyzer,
    llm_parser,
    enable_llm_routing=True,  # EXPLICITLY enable
)
results = router.analyze(conversation)
# ✓ Router decides when to use LLM
# ✓ Based on heuristics
# ✓ Optimal speed/accuracy trade-off
```

## Three Modes

### Mode 1: Hybrid Only (Default)

```python
router = AutoRouter(analyzer)
# LLM: DISABLED
# Speed: Instant (<0.1s)
# Cost: $0
# Use when: Speed matters, conversations are clear
```

### Mode 2: Intelligent Routing (Opt-In)

```python
router = AutoRouter(
    analyzer,
    llm_parser,
    enable_llm_routing=True,
)
# LLM: Conditionally used (heuristic-based)
# Speed: Variable (0.1s if skipped, 2-5s if used)
# Cost: Variable ($0 if skipped, ~$0.0001 if used)
# Use when: Want optimal balance, trust heuristics
```

### Mode 3: Always Use LLM (Manual)

```python
# Just call LLM parser directly
hybrid_results = analyzer.analyze(conversation)
llm_branches = parser.parse(conversation_text)
# LLM: Always used
# Speed: Slow (2-5s)
# Cost: Always ~$0.0001
# Use when: Need maximum accuracy, cost not a concern
```

## How Intelligent Routing Works

When `enable_llm_routing=True`, the router uses 5 heuristics:

1. **Complexity** (0-1) - Turn length, vocabulary
2. **Ambiguity** (0-1) - Hedging, conditionals
3. **Hybrid Quality** (0-1) - Branches found, confidence
4. **Coverage** (0-1) - % of turns with branches
5. **Implicit Signals** (0-1) - "Torn between", "depends"

**Decision Formula:**
```
Expected Value = 0.2×complexity + 0.3×ambiguity
                 - 0.25×quality - 0.15×coverage
                 + 0.3×implicit_signals

If Expected Value > Threshold → USE LLM
Else → SKIP LLM
```

## Configuration Strategies

### Balanced (Default)

```python
router = AutoRouter(
    analyzer,
    llm_parser,
    enable_llm_routing=True,
    router=IntelligentRouter(
        min_value_threshold=0.3,  # Use if 30%+ value
        prefer_speed=False,
    )
)
# Uses LLM when moderate value expected
```

### Speed-Optimized

```python
router = AutoRouter(
    analyzer,
    llm_parser,
    enable_llm_routing=True,
    router=IntelligentRouter(
        min_value_threshold=0.5,  # Use only if 50%+ value
        prefer_speed=True,
    )
)
# Biased toward hybrid, uses LLM rarely
```

### Thoroughness-Optimized

```python
router = AutoRouter(
    analyzer,
    llm_parser,
    enable_llm_routing=True,
    router=IntelligentRouter(
        min_value_threshold=0.1,  # Use if 10%+ value
        prefer_speed=False,
        time_budget_ms=30000,
    )
)
# Eager to use LLM, prioritizes accuracy
```

## Decision Examples

### Example 1: Clear Explicit Decision

**Input:**
```
BRANCH: Framework
1. Flask
2. FastAPI
```

**Heuristic Scores:**
- Complexity: 0.15 (short, simple)
- Hybrid Quality: 0.90 (explicit marker found)
- Expected Value: 0.15

**Decision:** SKIP LLM ✓ (hybrid perfect)

---

### Example 2: Ambiguous Implicit Decision

**Input:**
```
I'm torn between focusing on philosophy versus
practical advice. On one hand... on the other hand...
Perhaps it depends on the audience.
```

**Heuristic Scores:**
- Ambiguity: 0.70 ("torn between", "depends")
- Implicit Signals: 0.80 (lots of hedging)
- Hybrid Quality: 0.20 (found little)
- Expected Value: 0.65

**Decision:** USE LLM ✓ (high value expected)

---

### Example 3: Complex Conditional

**Input:**
```
If we go with microservices, we'd need Kubernetes,
but that depends on team expertise. If monolith,
then Django vs Rails, which depends on hiring...
```

**Heuristic Scores:**
- Complexity: 0.70 (nested logic)
- Implicit Signals: 0.90 (conditional chains)
- Expected Value: 0.72

**Decision:** USE LLM ✓ (complex reasoning)

## Verbose Mode

```python
results = router.analyze(conversation, verbose=True)
```

**Output when LLM disabled:**
```
🚀 LLM Routing: DISABLED (hybrid only mode)
   Set enable_llm_routing=True to enable intelligent LLM routing
```

**Output when LLM enabled:**
```
🤖 Router Decision: USE LLM
   Confidence: 0.85
   Reason: LLM recommended: high ambiguity, significant implicit signals (expected value: 0.65)
   Expected Value: 0.65

   Heuristic Scores:
     complexity: 0.45
     ambiguity: 0.70
     hybrid_quality: 0.20
     coverage: 0.40
     implicit_signals: 0.80

   LLM detected 2 additional branches
```

## Performance Comparison

| Configuration | Time | Cost | Branches | Use Case |
|---------------|------|------|----------|----------|
| Hybrid Only | 0.05s | $0 | 3 | Clear conversations |
| + Intelligent Router (skipped) | 0.05s | $0 | 3 | Good hybrid coverage |
| + Intelligent Router (used) | 2.5s | $0.0001 | 5 | Complex/ambiguous |
| Always LLM | 2.5s | $0.0001 | 5 | Maximum accuracy |

## Best Practices

### ✅ DO:

1. **Start with hybrid only** (default)
2. **Enable routing when needed** (complex conversations)
3. **Tune threshold** based on your speed/accuracy needs
4. **Use verbose mode** to understand decisions
5. **Add explicit markers** for critical decisions

### ❌ DON'T:

1. **Enable LLM by default** (wastes time/money)
2. **Ignore heuristic scores** (they're informative)
3. **Use always-LLM mode** unless necessary
4. **Forget to set budget constraints** (time/cost limits)

## Migration Guide

**Old approach (manual decision):**
```python
# You had to decide manually
if conversation_seems_complex:
    results = llm_parser.parse(conversation)
else:
    results = analyzer.analyze(conversation)
```

**New approach (intelligent routing):**
```python
# Router decides for you
router = AutoRouter(
    analyzer,
    llm_parser,
    enable_llm_routing=True,  # Opt-in
)
results = router.analyze(conversation)
```

## Demo

Run the full demonstration:

```bash
python examples/intelligent_routing_demo.py
```

This shows:
- Default hybrid-only behavior
- Intelligent routing with 4 test conversations
- Strategy comparison (balanced/speed/thoroughness)
- Heuristic scoring visualization

## Summary

**Key Principle:** LLM is **opt-in**, not automatic.

- **Default:** Fast, free hybrid detection
- **Opt-in:** Intelligent LLM routing when beneficial
- **Heuristic-based:** Automatic decision making
- **Configurable:** Tune to your preferences
- **Transparent:** Verbose mode explains decisions

The system respects your preference for speed and cost while giving you the option to enable deeper analysis when needed!
