"""
Example: Using Branch Detection Module

This demonstrates how to:
1. Extract branch points from text using deterministic patterns
2. Count maximum possible leaves
3. Get statistics about branch points
4. Optionally use LLM for complex cases
"""

from chatroutes_autobranch.branch_detection import (
    BranchExtractor,
    BranchPoint,
    LLMBranchParser,
)


def example_1_basic_extraction():
    """Example 1: Basic branch point extraction."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Branch Point Extraction")
    print("=" * 70)

    text = """
    For your backend, you have several options:
    1. Flask - lightweight and flexible
    2. FastAPI - modern and fast
    3. Django - batteries included

    The database choice depends on your needs. You can use Postgres or MySQL
    for relational data.

    If you need caching then use Redis else you can skip it.
    """

    extractor = BranchExtractor()
    branch_points = extractor.extract(text)

    print(f"\nText analyzed:")
    print(text)

    print(f"\n Found {len(branch_points)} branch points:")
    for i, bp in enumerate(branch_points, 1):
        print(f"\n{i}. Branch Point: {bp.id}")
        print(f"   Type: {bp.type}")
        print(f"   Options ({bp.option_count}):")
        for opt in bp.options:
            print(f"     - {opt.label}")

    # Calculate maximum leaves
    max_leaves = extractor.count_max_leaves(branch_points)
    print(f"\nMaximum possible combinations: {max_leaves}")
    print(f"  (3 backends × 2 databases × 2 caching = {3*2*2} paths)")


def example_2_statistics():
    """Example 2: Getting statistics about branch points."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Branch Point Statistics")
    print("=" * 70)

    text = """
    Hosting options:
    - Vercel
    - Fly.io
    - AWS
    - Heroku

    You can use Docker or native deployment.

    If production then use auto-scaling else use fixed instances.
    """

    extractor = BranchExtractor()
    branch_points = extractor.extract(text)

    stats = extractor.get_statistics(branch_points)

    print(f"\nStatistics:")
    print(f"  Total branch points:     {stats['total_branch_points']}")
    print(f"  Total options:           {stats['total_options']}")
    print(f"  Max possible leaves:     {stats['max_leaves']}")
    print(f"  Avg options per branch:  {stats['avg_options_per_branch']:.1f}")

    print(f"\n  By type:")
    for type_name, count in stats["by_type"].items():
        print(f"    {type_name}: {count}")


def example_3_real_world_llm_response():
    """Example 3: Analyzing a real LLM response."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Analyzing Real LLM Response")
    print("=" * 70)

    # Simulated LLM response
    llm_response = """
    Based on your requirements, here are the recommended approaches:

    **Backend Framework:**
    1. Flask (lightweight, good for small projects)
    2. FastAPI (modern, async support, fast)
    3. Django (full-featured, admin panel included)

    **Database Selection:**
    For your use case, either PostgreSQL or MySQL would work well.
    PostgreSQL offers better JSON support, while MySQL has wider adoption.

    **Deployment Strategy:**
    If you need zero-config deployment, use Vercel or Heroku.
    Alternatively, for more control, use AWS or GCP with Docker.

    **Caching:**
    - Redis (recommended for production)
    - Memcached (simpler alternative)
    - In-memory (for development only)
    """

    extractor = BranchExtractor()
    branch_points = extractor.extract(llm_response)

    print(f"LLM Response analyzed:")
    print(f"Found {len(branch_points)} decision points\n")

    for bp in branch_points:
        print(f"Decision: {bp.type.upper()}")
        print(f"  Options: {', '.join(bp.get_option_labels())}")
        print()

    max_combinations = extractor.count_max_leaves(branch_points)
    print(f"Total possible configuration combinations: {max_combinations}")


def example_4_empty_and_edge_cases():
    """Example 4: Handling edge cases."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Edge Cases")
    print("=" * 70)

    extractor = BranchExtractor()

    # Case 1: No branch points
    text1 = "This is just regular text with no options or choices."
    bp1 = extractor.extract(text1)
    print(f"Case 1 (no branches): Found {len(bp1)} branch points")

    # Case 2: Single item (not a branch)
    text2 = "Option: 1. Only one choice"
    bp2 = extractor.extract(text2)
    print(f"Case 2 (single item): Found {len(bp2)} branch points (ignored)")

    # Case 3: Multiple independent decisions
    text3 = """
    Color: red or blue or green
    Size: small or large
    """
    bp3 = extractor.extract(text3)
    print(f"Case 3 (2 decisions): Found {len(bp3)} branch points")
    print(f"  Max combinations: {extractor.count_max_leaves(bp3)} (3 colors × 2 sizes)")


def example_5_llm_parser_optional():
    """Example 5: Optional LLM parser for complex cases."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Optional LLM Parser (Demo)")
    print("=" * 70)

    # Mock LLM function (in production, this would call an actual LLM)
    def mock_llm(prompt: str) -> str:
        # Return mock JSON response
        return """{
            "branch_points": [
                {
                    "id": "bp1",
                    "type": "open_directive",
                    "options": [
                        {"id": "opt1", "label": "Microservices architecture", "span": "microservices"},
                        {"id": "opt2", "label": "Monolithic architecture", "span": "monolithic"}
                    ],
                    "context": "Choose an architecture pattern"
                }
            ]
        }"""

    text = "Choose an architecture pattern that fits your team size and complexity."

    # Try deterministic extractor first
    extractor = BranchExtractor()
    branch_points = extractor.extract(text)

    if len(branch_points) == 0:
        print("Deterministic extraction found no explicit options.")
        print("Trying LLM parser for implicit choices...")

        parser = LLMBranchParser(llm=mock_llm)
        branch_points, metadata = parser.parse_with_confidence(text)

        if metadata["success"]:
            print(f"\nLLM parser found {len(branch_points)} branch points:")
            for bp in branch_points:
                print(f"  - {bp.type}: {bp.get_option_labels()}")
        else:
            print(f"LLM parser failed: {metadata['error']}")
    else:
        print(f"Deterministic extraction found {len(branch_points)} branch points")


def main():
    """Run all examples."""
    example_1_basic_extraction()
    example_2_statistics()
    example_3_real_world_llm_response()
    example_4_empty_and_edge_cases()
    example_5_llm_parser_optional()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Branch Detection Module Usage:

1. BranchExtractor (deterministic):
   - Extracts enumerations (1. 2. 3. or - * •)
   - Detects disjunctions (A or B or C)
   - Finds conditionals (if...then...else)
   - Fast, no LLM needed

2. count_max_leaves():
   - Calculates product(k1 x k2 x ... x kn)
   - Shows maximum possible paths

3. get_statistics():
   - Total branch points
   - Options breakdown by type
   - Average options per branch

4. LLMBranchParser (optional):
   - For complex/implicit choices
   - Domain knowledge extraction
   - Fallback when patterns fail

Use deterministic first, LLM as fallback!
    """)


if __name__ == "__main__":
    main()
