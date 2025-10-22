"""
Basic usage example for chatroutes-autobranch.

This example demonstrates:
1. Creating a BranchSelector with all components
2. Running a single step() call
3. Inspecting the results
"""

from chatroutes_autobranch import (
    BranchSelector,
    BeamSelector,
    Budget,
    BudgetManager,
    Candidate,
    CosineNoveltyFilter,
    DummyEmbeddingProvider,
    ShannonEntropyStopper,
    StaticScorer,
)


def main():
    """Run basic usage example."""
    print("ChatRoutes AutoBranch - Basic Usage Example\n")

    # 1. Create embedding provider (use real embeddings in production)
    embedding_provider = DummyEmbeddingProvider(dimension=128, seed=42)

    # 2. Create scorer (use CompositeScorer in production)
    # StaticScorer assigns scores based on a dictionary mapping
    # For testing, we'll use a default dict that gives all candidates 0.75
    scorer = StaticScorer({})

    # 3. Create beam selector
    beam = BeamSelector(scorer=scorer, k=5)

    # 4. Create novelty filter
    novelty_filter = CosineNoveltyFilter(
        threshold=0.85, embedding_provider=embedding_provider
    )

    # 5. Create entropy stopper
    entropy_stopper = ShannonEntropyStopper(
        min_entropy=0.6, k_max=5, embedding_provider=embedding_provider
    )

    # 6. Create budget manager
    budget = Budget(max_nodes=32, max_tokens=30000, max_ms=12000)
    budget_manager = BudgetManager(budget, mode="strict")

    # 7. Create main branch selector
    selector = BranchSelector(
        beam_selector=beam,
        novelty_filter=novelty_filter,
        entropy_stopper=entropy_stopper,
        budget_manager=budget_manager,
    )

    # 8. Create parent and candidate nodes
    parent = Candidate(
        id="parent-1",
        text="What is the capital of France?",
        meta={"depth": 0},
    )

    candidates = [
        Candidate(id="c1", text="Paris is the capital of France."),
        Candidate(id="c2", text="The capital is Paris."),
        Candidate(id="c3", text="Lyon is a city in France."),
        Candidate(id="c4", text="France's capital is Paris."),
        Candidate(id="c5", text="Berlin is the capital of Germany."),
        Candidate(id="c6", text="Paris, the city of lights."),
        Candidate(id="c7", text="Marseille is in southern France."),
    ]

    # 9. Run selection step
    print(f"Parent: {parent.text}")
    print(f"Candidates: {len(candidates)}\n")

    result = selector.step(parent, candidates)

    # 10. Inspect results
    print("Results:")
    print(f"  Total scored: {len(result.scored)}")
    print(f"  After beam: {len(result.after_beam)}")
    print(f"  After novelty: {len(result.after_novelty)}")
    print(f"  Final kept: {len(result.kept)}")
    print(f"  Should continue: {result.should_continue}")

    print("\nKept candidates:")
    for i, cand in enumerate(result.kept, 1):
        print(f"  {i}. [{cand.id}] {cand.text[:50]}... (score: {cand.score:.3f})")

    print("\nBudget usage:")
    usage = budget_manager.usage
    print(f"  Nodes: {usage['used_nodes']}/{usage['max_nodes']}")
    print(f"  Tokens: {usage['used_tokens']}/{usage['max_tokens']}")
    print(f"  Latency: {usage['used_ms']}/{usage['max_ms']} ms")

    # 11. Demonstrate state management
    print("\nState management:")
    state = selector.get_state()
    print(f"  Saved state keys: {list(state.keys())}")

    # Reset for new tree
    selector.reset()
    print("  Selector reset for new tree")


if __name__ == "__main__":
    main()
