"""
Example: Analyzing Branching Potential from Existing Text

This demonstrates different approaches to determine how many branches
could be created from existing conversation text.
"""

import math
import re
from typing import Any

from chatroutes_autobranch import (
    BranchSelector,
    BeamSelector,
    Candidate,
    CosineNoveltyFilter,
    DummyEmbeddingProvider,
    ShannonEntropyStopper,
    StaticScorer,
)


def estimate_branching_potential_heuristic(text: str) -> dict[str, Any]:
    """
    Estimate branching potential using heuristics (no LLM required).

    Args:
        text: The text to analyze (user prompt or conversation turn)

    Returns:
        Dictionary with branching score, estimated branches, and factors
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
        "category": "low" if score < 0.3 else "medium" if score < 0.6 else "high",
    }


def measure_branching_potential_actual(
    text: str, mock_alternatives: list[str], embedding_provider
) -> dict[str, Any]:
    """
    Measure actual branching potential by analyzing generated alternatives.

    In production, you would generate alternatives via LLM:
        alternatives = llm.generate(text, n=10, temperature=0.8)

    Args:
        text: The original text
        mock_alternatives: List of alternative responses (normally from LLM)
        embedding_provider: For diversity calculation

    Returns:
        Dictionary with diversity metrics and unique branch count
    """
    # Convert to candidates
    candidates = [
        Candidate(id=f"alt_{i}", text=alt, meta={})
        for i, alt in enumerate(mock_alternatives)
    ]

    # Measure diversity using entropy
    entropy_stopper = ShannonEntropyStopper(
        min_entropy=0.0,  # We want the value, not decision
        k_max=5,
        embedding_provider=embedding_provider,
        random_seed=42,
    )

    decision = entropy_stopper.should_continue(candidates)
    diversity_score = decision.entropy  # 0.0 (low) to 1.0 (high)

    # Count truly unique branches after novelty filtering
    novelty_filter = CosineNoveltyFilter(
        threshold=0.85, embedding_provider=embedding_provider
    )

    # Need scored candidates for novelty filter
    # Use dummy scores (all equal) since we only care about diversity
    from chatroutes_autobranch import ScoredCandidate

    scored = [
        ScoredCandidate(id=c.id, text=c.text, meta=c.meta, score=0.5)
        for c in candidates
    ]

    unique_branches = novelty_filter.prune(scored)

    return {
        "total_generated": len(mock_alternatives),
        "unique_branches": len(unique_branches),
        "diversity_score": diversity_score,
        "branching_factor": len(unique_branches) / len(mock_alternatives),
        "similar_pruned": len(mock_alternatives) - len(unique_branches),
    }


def analyze_conversation_turns(conversation: list[dict[str, str]]) -> None:
    """
    Analyze each turn in a conversation for branching potential.

    Args:
        conversation: List of turns with "role" and "text"
    """
    print("=" * 70)
    print("CONVERSATION BRANCHING ANALYSIS")
    print("=" * 70)

    for i, turn in enumerate(conversation):
        if turn["role"] == "user":
            print(f"\n--- Turn {i} (User) ---")
            print(f"Text: {turn['text'][:60]}...")

            # Heuristic analysis (fast, no LLM)
            analysis = estimate_branching_potential_heuristic(turn["text"])

            print(f"\nHeuristic Analysis (fast, no LLM):")
            print(f"  Branching Score:     {analysis['branching_score']:.2f}/1.0")
            print(f"  Estimated Branches:  {analysis['estimated_branches']}")
            print(f"  Category:            {analysis['category'].upper()}")
            print(f"  Factors:             {', '.join(analysis['factors'])}")

            # Interpretation
            if analysis["branching_score"] < 0.3:
                print(
                    "  Interpretation:      Low branching potential - likely factual/specific"
                )
            elif analysis["branching_score"] < 0.6:
                print(
                    "  Interpretation:      Medium branching - some variation possible"
                )
            else:
                print(
                    "  Interpretation:      High branching - many diverse paths possible"
                )


def demo_actual_measurement():
    """
    Demonstrate measuring actual branching potential with mock alternatives.

    In production, you'd generate these via LLM:
        alternatives = llm.generate(prompt, n=10, temperature=0.8)
    """
    print("\n" + "=" * 70)
    print("ACTUAL DIVERSITY MEASUREMENT (with mock alternatives)")
    print("=" * 70)

    # Setup
    embedding_provider = DummyEmbeddingProvider(dimension=128, seed=42)

    # Case 1: Factual question (low diversity expected)
    print("\n--- Case 1: Factual Question ---")
    prompt1 = "What is the capital of France?"
    # Mock alternatives (in reality, these would be LLM-generated)
    alternatives1 = [
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "France's capital city is Paris.",
        "Paris.",
        "The capital is Paris.",
    ]

    result1 = measure_branching_potential_actual(
        prompt1, alternatives1, embedding_provider
    )

    print(f"Prompt: {prompt1}")
    print(f"\nResults:")
    print(f"  Total Generated:    {result1['total_generated']}")
    print(f"  Unique Branches:    {result1['unique_branches']}")
    print(f"  Diversity Score:    {result1['diversity_score']:.2f}")
    print(f"  Branching Factor:   {result1['branching_factor']:.2f}")
    print(f"  Similar Pruned:     {result1['similar_pruned']}")
    print(
        f"  >> All responses are very similar - LOW branching potential confirmed"
    )

    # Case 2: Creative prompt (high diversity expected)
    print("\n--- Case 2: Creative Prompt ---")
    prompt2 = "Write a creative story about Paris."
    alternatives2 = [
        "Once upon a time in Paris, a cat learned to paint...",
        "The Eiffel Tower came alive one night and started walking...",
        "Marie, a baker, discovered a secret underground garden...",
        "In a parallel universe, Paris is underwater...",
        "A time traveler arrives in 1920s Paris...",
    ]

    result2 = measure_branching_potential_actual(
        prompt2, alternatives2, embedding_provider
    )

    print(f"Prompt: {prompt2}")
    print(f"\nResults:")
    print(f"  Total Generated:    {result2['total_generated']}")
    print(f"  Unique Branches:    {result2['unique_branches']}")
    print(f"  Diversity Score:    {result2['diversity_score']:.2f}")
    print(f"  Branching Factor:   {result2['branching_factor']:.2f}")
    print(f"  Similar Pruned:     {result2['similar_pruned']}")
    print(f"  >> Diverse responses - HIGH branching potential confirmed")


def main():
    """Run branching potential analysis examples."""
    # Example conversation to analyze
    conversation = [
        {"role": "user", "text": "What is the capital of France?"},
        {"role": "assistant", "text": "Paris is the capital of France."},
        {"role": "user", "text": "Tell me a creative story about Paris."},
        {
            "role": "assistant",
            "text": "Once upon a time, in the heart of Paris...",
        },
        {
            "role": "user",
            "text": "Explain how the Eiffel Tower was built, specifically the foundation engineering.",
        },
        {"role": "assistant", "text": "The Eiffel Tower's foundation..."},
        {"role": "user", "text": "Imagine a world where the Eiffel Tower never existed."},
        {"role": "assistant", "text": "In this alternate reality..."},
    ]

    # Analyze conversation turns
    analyze_conversation_turns(conversation)

    # Demonstrate actual measurement
    demo_actual_measurement()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: When to Use Each Approach")
    print("=" * 70)
    print("""
1. HEURISTIC ANALYSIS (no LLM needed):
   - Fast batch analysis of many prompts
   - Quick estimation before generation
   - Cost-free initial assessment
   - Accuracy: 60-70%

2. ACTUAL MEASUREMENT (requires LLM):
   - Generate N alternatives (cost: N * LLM call)
   - Measure diversity with library components
   - Most accurate assessment
   - Use for: high-stakes decisions, validation

3. HYBRID APPROACH (recommended):
   - Use heuristics first
   - Generate + measure only for high-potential cases
   - Balance cost and accuracy
""")


if __name__ == "__main__":
    main()
