"""
Test script to verify determinism and reproducibility of chatroutes-autobranch.

This script runs the same pipeline multiple times with fixed seeds and verifies
that results are identical across runs.

Usage:
    python examples/test_reproducibility.py
"""

from chatroutes_autobranch import (
    BranchSelector,
    BeamSelector,
    CompositeScorer,
    MMRNoveltyFilter,
    ShannonEntropyStopper,
    DummyEmbeddingProvider,
    Candidate,
)


def create_deterministic_pipeline(seed: int = 42):
    """Create a fully deterministic pipeline with fixed seed."""
    # Deterministic embeddings
    embedding_provider = DummyEmbeddingProvider(dimension=64, seed=seed)

    # Deterministic scorer
    scorer = CompositeScorer(
        weights={"confidence": 0.4, "relevance": 0.4, "novelty": 0.2},
        embedding_provider=embedding_provider,
    )

    # Deterministic beam
    beam = BeamSelector(scorer=scorer, k=3)

    # Deterministic novelty filter
    novelty_filter = MMRNoveltyFilter(
        lambda_param=0.7, embedding_provider=embedding_provider
    )

    # Deterministic entropy stopper
    entropy_stopper = ShannonEntropyStopper(
        min_entropy=0.8,
        embedding_provider=embedding_provider,
        random_seed=seed,  # Fixed seed
    )

    # Complete deterministic pipeline
    selector = BranchSelector(
        beam_selector=beam,
        novelty_filter=novelty_filter,
        entropy_stopper=entropy_stopper,
    )

    return selector


def run_pipeline(selector, parent, candidates):
    """Run pipeline and return kept candidate IDs."""
    result = selector.step(parent, candidates)
    return [c.id for c in result.kept]


def test_reproducibility(num_runs: int = 10, seed: int = 42):
    """Test that pipeline produces identical results across multiple runs."""
    print("=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)
    print(f"\nTesting {num_runs} runs with seed={seed}")
    print(f"Expected: All runs produce IDENTICAL results\n")

    # Create test data
    parent = Candidate(
        id="prompt", text="Write a story about a detective", meta={"logprobs": -0.1}
    )

    candidates = [
        Candidate(
            id="c1",
            text="Detective Sarah Chen walked into the dimly lit office...",
            meta={"logprobs": -0.2},
        ),
        Candidate(
            id="c2",
            text="The rain hammered against the window as Detective Miller...",
            meta={"logprobs": -0.3},
        ),
        Candidate(
            id="c3",
            text="It was another cold morning when Detective Rodriguez...",
            meta={"logprobs": -0.25},
        ),
        Candidate(
            id="c4",
            text="Detective Park had seen many cases, but this one...",
            meta={"logprobs": -0.4},
        ),
        Candidate(
            id="c5",
            text="The phone rang at 3 AM. Detective Thompson knew...",
            meta={"logprobs": -0.35},
        ),
    ]

    # Run multiple times
    results = []
    for i in range(num_runs):
        # Create fresh pipeline with same seed
        selector = create_deterministic_pipeline(seed=seed)

        # Run pipeline
        kept_ids = run_pipeline(selector, parent, candidates)

        results.append(kept_ids)
        print(f"Run {i+1:2d}: {kept_ids}")

    # Verify all results are identical
    first_result = results[0]
    all_identical = all(result == first_result for result in results)

    print("\n" + "=" * 70)
    if all_identical:
        print("[OK] REPRODUCIBILITY VERIFIED!")
        print(f"[OK] All {num_runs} runs produced identical results")
        print(f"[OK] Kept candidates: {first_result}")
    else:
        print("[ERROR] REPRODUCIBILITY FAILED!")
        print("[ERROR] Results differ across runs")
        for i, result in enumerate(results):
            if result != first_result:
                print(f"[ERROR] Run {i+1} differs: {result}")

    print("=" * 70)

    return all_identical


def test_seed_variation():
    """Test that different seeds produce different results."""
    print("\n" + "=" * 70)
    print("SEED VARIATION TEST")
    print("=" * 70)
    print("\nTesting different seeds (should produce DIFFERENT results)\n")

    # Create test data
    parent = Candidate(
        id="prompt", text="Write a story about a detective", meta={"logprobs": -0.1}
    )

    candidates = [
        Candidate(id=f"c{i}", text=f"Response {i}", meta={"logprobs": -0.1 * i})
        for i in range(1, 11)
    ]

    # Test different seeds
    seeds = [42, 123, 999]
    results = {}

    for seed in seeds:
        selector = create_deterministic_pipeline(seed=seed)
        kept_ids = run_pipeline(selector, parent, candidates)
        results[seed] = kept_ids
        print(f"Seed {seed:3d}: {kept_ids}")

    # Verify results differ
    all_different = len(set(tuple(r) for r in results.values())) == len(seeds)

    print("\n" + "=" * 70)
    if all_different:
        print("[OK] SEED VARIATION VERIFIED!")
        print(f"[OK] Different seeds produced different results")
    else:
        print("[WARNING] Some seeds produced identical results")
        print("[WARNING] This may happen by chance with small candidate sets")

    print("=" * 70)

    return all_different


def main():
    print("\n")
    print("=" * 70)
    print("CHATROUTES-AUTOBRANCH REPRODUCIBILITY TEST SUITE")
    print("=" * 70)
    print("\nThis script verifies that the pipeline is fully deterministic")
    print("when using fixed seeds for all random components.\n")

    # Test 1: Reproducibility
    reproducible = test_reproducibility(num_runs=10, seed=42)

    # Test 2: Seed variation
    varies = test_seed_variation()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'[OK]' if reproducible else '[FAIL]'} Reproducibility: "
          f"Same seed -> Same results")
    print(f"{'[OK]' if varies else '[WARN]'} Seed variation: "
          f"Different seeds -> Different results")

    if reproducible and varies:
        print("\n[OK] All tests passed!")
        print("[OK] Pipeline is fully deterministic and reproducible")
        print("\nKey takeaways:")
        print("  1. Fixed seeds ensure 100% reproducibility")
        print("  2. Same inputs + same seeds -> identical outputs (every time)")
        print("  3. Different seeds -> different results (as expected)")
        print("\nFor reproducible research:")
        print("  - Always set seed in DummyEmbeddingProvider")
        print("  - Always set random_seed in ShannonEntropyStopper")
        print("  - Document seed values in your experiments")
        print("  - Use same model versions across runs")
    else:
        print("\n[FAIL] Some tests failed - reproducibility may be compromised")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
