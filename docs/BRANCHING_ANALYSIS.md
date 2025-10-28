# Branching Potential Analysis: Determining Branches from Existing Text

This document explains how to analyze existing conversations to determine:
1. **How many branches** could be created from a given point
2. **Which points** in a conversation have high branching potential
3. **Whether LLM help is needed** for this analysis

---

## Problem Statement

**Scenario**: You have an existing conversation:
```
User: "What's the capital of France?"
LLM: "Paris is the capital of France."

User: "Tell me more about it."
LLM: "Paris is known as the City of Light..."
```

**Questions**:
- At which points could we have taken different paths?
- How many viable alternative responses exist?
- How diverse would those alternatives be?

---

## Current Library Scope

⚠️ **Important**: The library currently assumes you **already have candidate branches** and helps you:
1. Score them
2. Select the best ones
3. Filter for diversity
4. Decide when to stop

It does **NOT** currently analyze existing text to determine branching potential.

---

## Approaches to Analyze Branching Potential

### **Approach 1: Generate-and-Measure** ⭐ (Recommended)

**Concept**: Actually generate N alternatives and measure their diversity.

**Algorithm**:
```python
def measure_branching_potential(text: str, llm, n: int = 10):
    """
    Generate N alternatives and measure how diverse they are.
    High diversity = high branching potential.
    """
    # 1. Generate N alternatives
    alternatives = llm.generate(text, n=n, temperature=0.8)

    # 2. Convert to candidates
    candidates = [Candidate(id=f"alt_{i}", text=alt) for i, alt in enumerate(alternatives)]

    # 3. Measure diversity using entropy
    entropy_stopper = ShannonEntropyStopper(
        min_entropy=0.0,  # We want the value, not decision
        k_max=5,
        embedding_provider=embedding_provider,
        random_seed=42
    )

    decision = entropy_stopper.should_continue(candidates)
    diversity_score = decision.entropy  # 0.0 (low) to 1.0 (high)

    # 4. Count truly unique branches after novelty filtering
    novelty_filter = CosineNoveltyFilter(
        threshold=0.85,
        embedding_provider=embedding_provider
    )
    unique_branches = novelty_filter.prune(candidates)

    return {
        "total_generated": n,
        "unique_branches": len(unique_branches),
        "diversity_score": diversity_score,
        "branching_factor": len(unique_branches) / n,  # 0.0 to 1.0
    }
```

**Pros**:
- ✅ Accurate measure of actual branching potential
- ✅ Uses existing library components (entropy, novelty)
- ✅ Gives quantitative metrics

**Cons**:
- ❌ Requires LLM generation (cost + latency)
- ❌ Not applicable if you want to analyze without generation

**Use When**: You want accurate branching potential and can afford generation costs.

---

### **Approach 2: Heuristic Analysis** (No LLM Required)

**Concept**: Analyze text characteristics to estimate branching potential.

**Heuristics**:
```python
def estimate_branching_potential(text: str, previous_context: str = "") -> dict:
    """
    Estimate branching potential using heuristics (no LLM needed).
    """
    score = 0.5  # Base score
    factors = []

    # 1. Question detection (high branching)
    if any(q in text for q in ["?", "how", "why", "what", "explain"]):
        score += 0.3
        factors.append("contains_question")

    # 2. Open-ended prompts (high branching)
    open_ended_keywords = ["tell me about", "describe", "imagine", "create", "write"]
    if any(keyword in text.lower() for keyword in open_ended_keywords):
        score += 0.2
        factors.append("open_ended")

    # 3. Specificity (low branching)
    specific_patterns = [
        r"\d+",  # Contains numbers
        r"specific|exact|precise",
        r"yes or no",
    ]
    import re
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in specific_patterns):
        score -= 0.2
        factors.append("specific_question")

    # 4. Factual questions (low branching)
    factual_keywords = ["capital", "when was", "who is", "what is the"]
    if any(keyword in text.lower() for keyword in factual_keywords):
        score -= 0.15
        factors.append("factual")

    # 5. Creative prompts (very high branching)
    creative_keywords = ["creative", "story", "poem", "imagine", "invent"]
    if any(keyword in text.lower() for keyword in creative_keywords):
        score += 0.4
        factors.append("creative")

    # 6. Length (longer = more constraint = lower branching)
    if len(text) > 500:
        score -= 0.1
        factors.append("long_prompt")

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    # Estimate branch count
    # 0.0-0.3: Low (1-2 branches)
    # 0.3-0.6: Medium (2-4 branches)
    # 0.6-1.0: High (4-8+ branches)
    estimated_branches = int(1 + score * 7)

    return {
        "branching_score": score,
        "estimated_branches": estimated_branches,
        "factors": factors,
        "category": "low" if score < 0.3 else "medium" if score < 0.6 else "high"
    }
```

**Pros**:
- ✅ Fast (no LLM calls)
- ✅ Deterministic
- ✅ No API costs

**Cons**:
- ❌ Less accurate (heuristics vs actual generation)
- ❌ Misses semantic nuances
- ❌ Requires manual tuning of heuristics

**Use When**: Quick analysis, no budget for generation, or batch analysis of large datasets.

---

### **Approach 3: Model Confidence Analysis** (Requires Logprobs)

**Concept**: Use model confidence/uncertainty as proxy for branching potential.

**Algorithm**:
```python
def analyze_via_confidence(text: str, llm_response: dict):
    """
    Use model's confidence (logprobs) to estimate branching potential.
    Low confidence = high uncertainty = more branching potential.
    """
    # Assumes llm_response contains logprobs
    avg_logprob = llm_response.get("logprobs", {}).get("avg_logprob", -0.5)

    # Convert logprob to confidence (exponential)
    # logprob ranges: -0.1 (very confident) to -3.0 (uncertain)
    confidence = math.exp(avg_logprob)  # 0.0 to 1.0

    # Invert: high confidence = low branching potential
    branching_potential = 1.0 - confidence

    # Check for multiple high-probability tokens at key positions
    token_entropy = calculate_token_entropy(llm_response.get("logprobs", {}).get("token_logprobs", []))

    return {
        "branching_potential": branching_potential,
        "confidence": confidence,
        "token_entropy": token_entropy,
        "estimated_branches": int(1 + branching_potential * 6)
    }

def calculate_token_entropy(token_logprobs: list) -> float:
    """Calculate average entropy across tokens."""
    if not token_logprobs:
        return 0.5

    entropies = []
    for logprob in token_logprobs:
        # If we have top-k logprobs, calculate entropy
        if isinstance(logprob, dict):
            probs = [math.exp(lp) for lp in logprob.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            entropies.append(entropy)

    return sum(entropies) / len(entropies) if entropies else 0.5
```

**Pros**:
- ✅ Uses signal from actual model
- ✅ No additional generation needed (if you already have response)
- ✅ Theoretically grounded (uncertainty → diversity)

**Cons**:
- ❌ Requires logprobs (not all APIs provide them)
- ❌ Only works for existing responses (not for analyzing prompts)
- ❌ Correlation isn't perfect (high confidence ≠ low branching always)

**Use When**: You have existing LLM responses with logprobs and want to retroactively analyze them.

---

### **Approach 4: LLM-as-Judge Meta-Analysis**

**Concept**: Ask an LLM to analyze branching potential.

**Algorithm**:
```python
def llm_analyze_branching(text: str, llm) -> dict:
    """
    Use LLM to analyze branching potential metacognitively.
    """
    analysis_prompt = f"""
Analyze this text and determine its branching potential:

Text: "{text}"

Questions:
1. How many significantly different valid responses could exist? (1-10)
2. What is the diversity potential? (low/medium/high)
3. What are the main branching dimensions (e.g., tone, depth, perspective)?

Respond in JSON format:
{{
    "estimated_branches": <number>,
    "diversity": "<low|medium|high>",
    "dimensions": ["dimension1", "dimension2"],
    "reasoning": "brief explanation"
}}
"""

    response = llm.generate(analysis_prompt, temperature=0.1)  # Low temp for consistency
    import json
    result = json.loads(response)

    return result
```

**Pros**:
- ✅ Understands semantic context
- ✅ Can explain reasoning
- ✅ Handles nuanced cases

**Cons**:
- ❌ Expensive (LLM call per analysis)
- ❌ Not deterministic (model variation)
- ❌ May hallucinate or be inconsistent

**Use When**: You need semantic understanding and can afford the cost.

---

### **Approach 5: Retrospective Branching** (Conversation Tree Reconstruction)

**Concept**: Given a conversation, retroactively generate alternative branches at each turn.

**Algorithm**:
```python
def analyze_conversation_branching(conversation: list[dict], llm, selector: BranchSelector):
    """
    Analyze each turn in a conversation for branching potential.

    Args:
        conversation: [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}]
    """
    analysis = []

    for i, turn in enumerate(conversation):
        if turn["role"] != "assistant":
            continue

        # Get context (previous turns)
        context = conversation[:i]
        user_prompt = conversation[i-1]["text"] if i > 0 else ""

        # Create parent candidate
        parent = Candidate(
            id=f"turn_{i-1}",
            text=user_prompt,
            meta={"turn": i-1}
        )

        # Generate N alternatives
        alternatives = llm.generate(user_prompt, n=5, temperature=0.8, context=context)
        candidates = [
            Candidate(id=f"turn_{i}_alt_{j}", text=alt)
            for j, alt in enumerate(alternatives)
        ]

        # Add actual response
        actual = Candidate(id=f"turn_{i}_actual", text=turn["text"])
        candidates.append(actual)

        # Analyze using selector
        result = selector.step(parent, candidates)

        analysis.append({
            "turn": i,
            "user_prompt": user_prompt,
            "actual_response": turn["text"],
            "alternatives_generated": len(alternatives),
            "unique_branches": len(result.kept),
            "entropy": result.entropy_decision.get("entropy", 0),
            "diversity_score": result.entropy_decision.get("entropy", 0),
            "kept_branches": [c.id for c in result.kept],
            "was_actual_kept": actual.id in [c.id for c in result.kept]
        })

    return analysis
```

**Pros**:
- ✅ Complete analysis of entire conversation
- ✅ Uses full library capabilities
- ✅ Can identify which actual responses were "good" (kept)

**Cons**:
- ❌ Expensive (N generations per turn)
- ❌ Requires generation capability

**Use When**: Comprehensive conversation analysis, quality assessment, or training data generation.

---

## Practical Decision Matrix

| Use Case | Approach | LLM Needed? | Cost | Accuracy |
|----------|----------|-------------|------|----------|
| **Quick estimation** | Heuristics (#2) | No | Free | Low-Medium |
| **Existing responses analysis** | Confidence (#3) | No* | Free | Medium |
| **Accurate potential** | Generate-and-Measure (#1) | Yes | High | High |
| **Semantic understanding** | LLM-as-Judge (#4) | Yes | Medium | Medium-High |
| **Full conversation audit** | Retrospective (#5) | Yes | Very High | Very High |

*Requires existing responses with logprobs

---

## Recommended Implementation

**For Most Use Cases**: Hybrid approach

```python
def analyze_branching_potential(text: str, mode: str = "fast"):
    """
    Analyze branching potential with configurable detail level.
    """
    if mode == "fast":
        # Heuristic analysis (no LLM)
        return estimate_branching_potential(text)

    elif mode == "accurate":
        # Generate and measure (requires LLM)
        return measure_branching_potential(text, llm, n=10)

    elif mode == "hybrid":
        # Quick heuristic, then validate with sampling
        heuristic = estimate_branching_potential(text)

        if heuristic["branching_score"] > 0.6:  # High potential
            # Confirm with actual generation (sample N=5)
            actual = measure_branching_potential(text, llm, n=5)
            return actual
        else:
            # Trust heuristic for low potential
            return heuristic
```

---

## Example: Analyzing a Conversation

```python
from chatroutes_autobranch import BranchSelector, Candidate

# Your conversation
conversation = [
    {"role": "user", "text": "What's the capital of France?"},
    {"role": "assistant", "text": "Paris is the capital of France."},
    {"role": "user", "text": "Tell me a creative story about it."},
    {"role": "assistant", "text": "Once upon a time in Paris..."},
]

# Analyze each turn
for i, turn in enumerate(conversation):
    if turn["role"] == "user":
        # Heuristic analysis (fast)
        analysis = estimate_branching_potential(turn["text"])

        print(f"\nTurn {i}: {turn['text'][:50]}...")
        print(f"  Branching Score: {analysis['branching_score']:.2f}")
        print(f"  Estimated Branches: {analysis['estimated_branches']}")
        print(f"  Category: {analysis['category']}")
        print(f"  Factors: {', '.join(analysis['factors'])}")

# Output:
# Turn 0: What's the capital of France?...
#   Branching Score: 0.35
#   Estimated Branches: 3
#   Category: medium
#   Factors: contains_question, factual

# Turn 2: Tell me a creative story about it....
#   Branching Score: 0.85
#   Estimated Branches: 6
#   Category: high
#   Factors: contains_question, open_ended, creative
```

---

## Summary

**Do you need LLM help?**

- **For quick estimation**: No (use heuristics)
- **For accurate measurement**: Yes (generate alternatives)
- **For existing conversations**: Maybe (depends on whether you have logprobs)

**Recommended Strategy**:
1. Start with heuristics for batch analysis
2. Use generation for high-potential cases
3. Use retrospective analysis for quality assessment

**Key Insight**: Branching potential is best measured by **actual diversity** of generated alternatives, but can be reasonably estimated with heuristics for most cases.
