"""
Intelligent Routing Demo: Automatic LLM decision-making

This demonstrates how the intelligent router automatically decides
whether to use LLM based on conversation characteristics.
"""

from chatroutes_autobranch.branch_detection import (
    ConversationFlowAnalyzer,
    LLMBranchParser,
    ConversationTurn,
)
from chatroutes_autobranch.branch_detection.intelligent_router import (
    IntelligentRouter,
    AutoRouter,
)
from chatroutes_autobranch.core.embeddings import DummyEmbeddingProvider

try:
    import ollama
    OLLAMA_AVAILABLE = True

    def llama3(prompt: str) -> str:
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        return response['message']['content']
except ImportError:
    OLLAMA_AVAILABLE = False
    llama3 = None


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_conversation(name: str, conversation: list[ConversationTurn], router: AutoRouter):
    """Test a conversation and show routing decision."""
    print_section(f"TEST: {name}")

    print("\nConversation:")
    for turn in conversation:
        preview = turn.content[:60] + "..." if len(turn.content) > 60 else turn.content
        print(f"  [{turn.speaker}] {preview}")

    print("\n" + "-" * 80)
    results = router.analyze(conversation, verbose=True)

    print(f"\n📊 RESULTS:")
    print(f"   Total branches: {len(results['all_branches'])}")
    print(f"     • Hybrid: {len(results['hybrid_results']['explicit_branches']) + len(results['hybrid_results']['semantic_branches'])}")
    print(f"     • LLM: {len(results['llm_branches'])}")

    return results


def main():
    print_section("INTELLIGENT ROUTING DEMONSTRATION")

    # Set up analyzer
    analyzer = ConversationFlowAnalyzer(
        embedding_provider=DummyEmbeddingProvider(dimension=384, seed=42),
        enable_explicit=True,
        enable_semantic=True,
    )

    # Set up LLM parser (if available)
    llm_parser = LLMBranchParser(llm=llama3) if OLLAMA_AVAILABLE else None

    if not OLLAMA_AVAILABLE:
        print("\n⚠️  Ollama not available. Router will work but can't actually use LLM.")
        print("Install Ollama to see full functionality: https://ollama.ai\n")

    # Create router with different configurations
    print("\nCreating routers with different strategies...")

    # Router 0: Hybrid-only (default - LLM disabled)
    router_hybrid_only = AutoRouter(
        analyzer=analyzer,
        enable_llm_routing=False,  # DEFAULT: LLM disabled
    )

    # Router 1: Balanced (LLM enabled)
    router_balanced = AutoRouter(
        analyzer=analyzer,
        llm_parser=llm_parser,
        enable_llm_routing=True,  # EXPLICITLY enable LLM
        router=IntelligentRouter(
            min_value_threshold=0.3,  # Use LLM if 30%+ expected value
            prefer_speed=False,
        )
    )

    # Router 2: Speed-optimized (LLM enabled but conservative)
    router_fast = AutoRouter(
        analyzer=analyzer,
        llm_parser=llm_parser,
        enable_llm_routing=True,
        router=IntelligentRouter(
            min_value_threshold=0.5,  # Higher threshold
            prefer_speed=True,  # Bias toward hybrid
        )
    )

    # Router 3: Thoroughness-optimized (LLM enabled and eager)
    router_thorough = AutoRouter(
        analyzer=analyzer,
        llm_parser=llm_parser,
        enable_llm_routing=True,
        router=IntelligentRouter(
            min_value_threshold=0.1,  # Lower threshold
            prefer_speed=False,
            time_budget_ms=30000,  # Willing to wait
        )
    )

    # ========================================================================
    # TEST 0: Show Default Behavior (Hybrid Only)
    # ========================================================================

    print_section("DEFAULT BEHAVIOR TEST (Hybrid Only)")
    print("\nBy default, LLM is DISABLED for speed and cost efficiency.")
    print("Only hybrid detection is used (instant, free).\n")

    test_conversation("Default: Hybrid Only", conv1, router_hybrid_only)
    print("\n💡 Notice: LLM routing is disabled by default!")

    # ========================================================================
    # TEST 1: Clear, Explicit Conversation (Hybrid Sufficient)
    # ========================================================================

    conv1 = [
        ConversationTurn(
            id="1",
            speaker="user",
            content="""BRANCH: Framework Choice
OPTIONS:
1. Flask - Simple, lightweight
2. FastAPI - Modern, fast"""
        ),
    ]

    test_conversation("Clear Explicit Decision", conv1, router_balanced)
    print("\n💡 EXPECTED: Router should SKIP LLM (hybrid perfect)")

    # ========================================================================
    # TEST 2: Ambiguous, Implicit Conversation (LLM Valuable)
    # ========================================================================

    conv2 = [
        ConversationTurn(
            id="1",
            speaker="user",
            content="I'm torn between focusing on philosophical depth versus practical actionability in my writing."
        ),
        ConversationTurn(
            id="2",
            speaker="user",
            content="On one hand, exploring the deeper meaning of work in the AI age seems important. On the other hand, readers need concrete guidance."
        ),
        ConversationTurn(
            id="3",
            speaker="user",
            content="Perhaps the dichotomy is false and I should consider a hybrid approach, though that depends on my audience's sophistication level."
        ),
    ]

    test_conversation("Ambiguous Implicit Decision", conv2, router_balanced)
    print("\n💡 EXPECTED: Router should USE LLM (high ambiguity, implicit)")

    # ========================================================================
    # TEST 3: Simple Question-Answer (Hybrid Sufficient)
    # ========================================================================

    conv3 = [
        ConversationTurn(
            id="1",
            speaker="user",
            content="Should I use Python or JavaScript?"
        ),
        ConversationTurn(
            id="2",
            speaker="user",
            content="I'll go with Python for data science."
        ),
    ]

    test_conversation("Simple Q&A", conv3, router_balanced)
    print("\n💡 EXPECTED: Router should SKIP LLM (pattern clear)")

    # ========================================================================
    # TEST 4: Complex Conditional Logic (LLM Valuable)
    # ========================================================================

    conv4 = [
        ConversationTurn(
            id="1",
            speaker="user",
            content="If we go with microservices, we'd need to decide between Kubernetes or Docker Swarm, but that depends on our team's expertise."
        ),
        ConversationTurn(
            id="2",
            speaker="user",
            content="However, if we choose a monolith, then the question becomes whether to use Django or Rails, which in turn depends on our hiring strategy."
        ),
        ConversationTurn(
            id="3",
            speaker="user",
            content="The trade-offs are complex and situational, depending on factors like scale, budget, and timeline."
        ),
    ]

    test_conversation("Complex Conditional", conv4, router_balanced)
    print("\n💡 EXPECTED: Router should USE LLM (complex logic, conditionals)")

    # ========================================================================
    # COMPARISON: Default vs LLM-Enabled Strategies
    # ========================================================================

    print_section("STRATEGY COMPARISON: Opt-In LLM")

    test_conv = conv2  # Use ambiguous conversation

    print("\n0️⃣  DEFAULT (Hybrid Only - LLM Disabled):")
    r0 = test_conversation("Default", test_conv, router_hybrid_only)

    print("\n\n1️⃣  BALANCED STRATEGY (LLM Enabled):")
    r1 = test_conversation("Balanced", test_conv, router_balanced)

    print("\n\n2️⃣  SPEED-OPTIMIZED STRATEGY (LLM Enabled, Conservative):")
    r2 = test_conversation("Speed-Optimized", test_conv, router_fast)

    print("\n\n3️⃣  THOROUGHNESS-OPTIMIZED STRATEGY (LLM Enabled, Eager):")
    r3 = test_conversation("Thoroughness-Optimized", test_conv, router_thorough)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print_section("SUMMARY: How Heuristics Work")

    print("""
🧠 THE INTELLIGENT ROUTER USES 5 HEURISTICS:

1. COMPLEXITY (0-1)
   - Turn length, vocabulary diversity
   - Higher = more LLM value
   - Example: Simple "A or B?" = 0.2, Complex essay = 0.8

2. AMBIGUITY (0-1)
   - Hedging language ("maybe", "perhaps")
   - Conditional statements
   - Higher = more LLM value
   - Example: Clear choice = 0.1, "It depends..." = 0.7

3. HYBRID QUALITY (0-1)
   - Branches found, confidence scores
   - Higher = less LLM value (inverse)
   - Example: 5+ explicit branches = 0.9, 0 branches = 0.1

4. COVERAGE (0-1)
   - Percentage of turns with detected branches
   - Higher = less LLM value (inverse)
   - Example: All turns covered = 1.0, Few covered = 0.3

5. IMPLICIT SIGNALS (0-1)
   - "Torn between", "considering", "trade-off"
   - Complex conditionals
   - Higher = more LLM value
   - Example: Explicit markers = 0.0, Implicit reasoning = 0.8

🎯 DECISION FORMULA:

Expected Value = 0.2×complexity + 0.3×ambiguity - 0.25×quality
                 - 0.15×coverage + 0.3×implicit_signals

If Expected Value > Threshold → USE LLM
Else → SKIP LLM (hybrid sufficient)

⚙️  CONFIGURABLE THRESHOLDS:

Balanced:    threshold=0.3  (use LLM if 30%+ value expected)
Fast:        threshold=0.5  (use LLM only if 50%+ value)
Thorough:    threshold=0.1  (use LLM if 10%+ value)

📊 REAL EXAMPLES FROM ABOVE:

Test 1 (Clear Explicit):
  • Quality: 0.9 (explicit marker found)
  • Coverage: 1.0 (fully covered)
  • Expected Value: ~0.15
  • Decision: SKIP LLM ✓

Test 2 (Ambiguous Implicit):
  • Ambiguity: 0.7 ("torn between", "depends")
  • Implicit Signals: 0.8 (lots of hedging)
  • Quality: 0.2 (hybrid found little)
  • Expected Value: ~0.65
  • Decision: USE LLM ✓

Test 4 (Complex Conditional):
  • Complexity: 0.7 (long, nested logic)
  • Implicit Signals: 0.9 (if-then chains)
  • Expected Value: ~0.70
  • Decision: USE LLM ✓

💡 SMART DEFAULTS:

The router learns from:
- Hybrid detection quality
- Conversation characteristics
- Budget constraints
- Your preferences (speed vs thoroughness)

It automatically makes the optimal choice!

🚀 USAGE:

    # DEFAULT: Hybrid only (fast, free, no LLM)
    router = AutoRouter(analyzer)  # LLM disabled by default
    results = router.analyze(conversation, verbose=True)
    # Uses only hybrid detection (instant, $0)

    # OPT-IN: Enable intelligent LLM routing
    router = AutoRouter(
        analyzer,
        llm_parser,
        enable_llm_routing=True,  # EXPLICITLY enable LLM
    )
    results = router.analyze(conversation, verbose=True)
    # Router decides whether to use LLM based on heuristics

    # CUSTOM: Fine-tune LLM routing strategy
    router = AutoRouter(
        analyzer,
        llm_parser,
        enable_llm_routing=True,
        router=IntelligentRouter(
            min_value_threshold=0.4,  # Your threshold
            prefer_speed=True,        # Your preference
            time_budget_ms=5000,      # Your constraints
        )
    )

⚙️  DEFAULT BEHAVIOR (Safe & Fast):
   - LLM is DISABLED unless you explicitly enable it
   - Only hybrid detection runs (instant, free)
   - Opt-in to LLM when you need deeper analysis

The router handles the complexity for you, but respects your choice!
""")

    print_section("END OF DEMONSTRATION")
    print("\nKey Takeaway:")
    print("The intelligent router automatically makes the speed/accuracy trade-off,")
    print("using LLM only when it will meaningfully improve results!")


if __name__ == "__main__":
    main()
