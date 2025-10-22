"""
Creative Writing Usage Example with 100% FREE Models.

This example demonstrates chatroutes-autobranch for creative writing scenarios:
1. Ollama models (qwen3:14b, gpt-oss:20b, or llama3.1:8b) for candidates (100% FREE!)
2. FREE sentence-transformers embeddings for semantic comparison (no cost!)
3. High temperature generation for diversity
4. Entropy-based stopping for convergence detection
5. Novelty filtering to avoid repetition

Requirements:
    pip install sentence-transformers requests

    # Install Ollama:
    # 1. Download from https://ollama.ai
    # 2. Install ONE of these models:
    #    - RECOMMENDED: ollama pull qwen3:14b (14B, fast, excellent quality)
    #    - ALTERNATIVE: ollama pull gpt-oss:20b (20B, OpenAI model, slower)
    #    - FALLBACK:    ollama pull llama3.1:8b (8B, fast, good baseline)
    # 3. Ollama runs in background automatically

100% FREE Strategy:
    - Use Ollama (any model above) for text generation (FREE, runs locally!)
    - Use sentence-transformers embeddings (FREE, runs locally!)
    - Total cost: $0 - Everything runs on your machine!

Creative Writing Scenarios:
1. AI Memory Story - Shows entropy staying high across diverse genres
   Uses: jina-embeddings-v2-base-en (768D, score: 60.3, excellent quality)

2. Mars Detective Twists - Demonstrates clustering of distinct ideas
   Uses: all-mpnet-base-v2 (768D, score: 57.8, good quality, faster)

3. Rom-Com Endings - Perfect entropy-based stopping visualization
   Uses: BAAI/bge-large-en-v1.5 (1024D, score: 59.5, very good quality)

4. Style Variations - Intent alignment and style variance
   Uses: jina-embeddings-v2-base-en (768D, score: 60.3, excellent quality)

Note: Each scenario uses a different free embedding model to demonstrate variety.
      All models are excellent quality (scores 57.8-60.3, nearly as good as OpenAI 64.6).
"""

import sys
import os
import requests
import time
from datetime import datetime
from typing import Optional

# Force UTF-8 output for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers package not installed.")
    print("Install with: pip install sentence-transformers")
    exit(1)

from chatroutes_autobranch import (
    BranchSelector,
    BeamSelector,
    Budget,
    BudgetManager,
    Candidate,
    CompositeScorer,
    CosineNoveltyFilter,
    MMRNoveltyFilter,
    ShannonEntropyStopper,
)


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def get_timestamp() -> str:
    """Get current timestamp in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")


class SentenceTransformerEmbeddingProvider:
    """
    FREE embedding provider using sentence-transformers.

    Implements the EmbeddingProvider protocol for chatroutes-autobranch.
    Uses open-source models - no API costs!

    Recommended models:
        - jinaai/jina-embeddings-v2-base-en: 768D, excellent quality (score: 60.3)
        - BAAI/bge-large-en-v1.5: 1024D, very good quality (score: 59.5)
        - all-mpnet-base-v2: 768D, good quality, fast (score: 57.8)
        - all-MiniLM-L6-v2: 384D, decent quality, very fast (score: 53.9)
    """

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-en"):
        """
        Initialize sentence-transformers embedding provider.

        Args:
            model_name: Name of the sentence-transformers model.
                       First run will download the model (~400MB).
        """
        print(f"[FREE Embeddings] Loading model: {model_name}")
        print(f"[FREE Embeddings] First run will download model (~400MB)")
        self.model = SentenceTransformer(model_name)
        print(f"[FREE Embeddings] Model loaded successfully!")
        print(f"[FREE Embeddings] Dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (normalized to unit length).
        """
        if not texts:
            return []

        try:
            # encode() returns numpy array, convert to list of lists
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            embeddings_list = embeddings.tolist()
            print(f"[FREE Embeddings] Generated {len(embeddings_list)} embeddings (FREE!)")
            return embeddings_list
        except Exception as e:
            print(f"[FREE Embeddings] ERROR: {e}")
            raise


def check_ollama_available() -> tuple[bool, str]:
    """
    Check if Ollama is running and which model is available.

    Returns:
        Tuple of (is_available, model_name)
        - Prefers qwen3:14b (best quality, newest)
        - Falls back to gpt-oss:20b (OpenAI open model, excellent)
        - Falls back to llama3.1:8b (good baseline)
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check for llama3.1:8b first (fastest for testing)
            if any("llama3.1:8b" in name or "llama3.1" in name for name in model_names):
                print("[OK] Found llama3.1:8b (FAST for testing!)")
                print("[INFO] For best quality (slower): qwen3:14b, gpt-oss:20b")
                return True, "llama3.1:8b"

            # Check for qwen3:14b (better quality, slower on CPU)
            if any("qwen3:14b" in name for name in model_names):
                print("[OK] Found qwen3:14b (latest, best quality, slow on CPU)")
                return True, "qwen3:14b"

            # Check for gpt-oss:20b (OpenAI's open model - excellent!)
            if any("gpt-oss:20b" in name or "gpt-oss" in name for name in model_names):
                print("[OK] Found gpt-oss:20b (OpenAI open model, excellent quality!)")
                return True, "gpt-oss:20b"

            # No suitable model found
            print("\nERROR: No suitable model found!")
            print("Please install one of these models:")
            print("  - RECOMMENDED: ollama pull qwen3:14b (14B, latest, best quality)")
            print("  - ALTERNATIVE: ollama pull gpt-oss:20b (20B, OpenAI open model)")
            print("  - FALLBACK:    ollama pull llama3.1:8b (8B, good baseline)")
            return False, ""

        return False, ""
    except requests.exceptions.RequestException:
        print("\nERROR: Ollama is not running!")
        print("Please start Ollama:")
        print("  - Windows: Ollama should auto-start, or run from Start Menu")
        print("  - Mac/Linux: ollama serve")
        return False, ""


def generate_creative_candidates(
    prompt: str,
    model_name: str = "qwen3:14b",
    n: int = 10,
    temperature: float = 1.2,
    parent_id: str = "parent-1"
) -> tuple[Candidate, list[Candidate]]:
    """
    Generate creative writing candidates using Ollama (FREE!).

    Args:
        prompt: The creative writing prompt.
        model_name: Name of the Ollama model to use (e.g., "qwen3:14b", "llama3.1:8b").
        n: Number of candidates to generate.
        temperature: Sampling temperature (higher = more creative).
                    0.0 = deterministic, 2.0 = very creative
        parent_id: ID for the parent candidate.

    Returns:
        Tuple of (parent, candidates).
    """
    start_time = time.time()
    print(f"\n[{get_timestamp()}] [Ollama {model_name}] Generating {n} creative responses...")
    print(f"[Ollama] Temperature: {temperature} (high for diversity)")
    print(f"[Ollama] Model: {model_name} (FREE, runs locally!)")

    # Create parent candidate
    parent = Candidate(
        id=parent_id,
        text=prompt,
        meta={
            "intent": "creative_writing",
            "depth": 0,
            "temperature": temperature
        }
    )

    # Generate multiple creative responses
    candidates = []
    system_prompt = "You are a creative writing assistant. Generate diverse, imaginative responses."

    try:
        # qwen3 and gpt-oss have "thinking" phase, need longer timeout
        timeout = 180 if ("gpt-oss" in model_name or "qwen3" in model_name) else 60

        for i in range(n):
            candidate_start = time.time()

            # Ollama API call
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "num_predict": 500,  # max tokens
                        "temperature": temperature
                    }
                },
                timeout=timeout
            )

            candidate_elapsed = time.time() - candidate_start

            if response.status_code != 200:
                print(f"[{get_timestamp()}] [Ollama] ERROR: HTTP {response.status_code}")
                continue

            data = response.json()
            text = data.get("response", "").strip()

            if not text:
                print(f"[{get_timestamp()}] [Ollama] WARNING: Empty response for candidate {i+1} (took {format_elapsed_time(candidate_elapsed)})")
                continue

            candidate = Candidate(
                id=f"creative-{i+1:03d}",
                text=text,
                meta={
                    "logprobs": -0.5,  # Neutral default
                    "target_intent": "creative_writing",
                    "model": model_name,
                    "temperature": temperature,
                    "parent_id": parent_id
                }
            )
            candidates.append(candidate)

            # Show progress with timing
            if (i + 1) % 3 == 0 or i == n - 1:
                avg_time = (time.time() - start_time) / (i + 1)
                print(f"[{get_timestamp()}] Progress: {i+1}/{n} candidates | Last: {format_elapsed_time(candidate_elapsed)} | Avg: {format_elapsed_time(avg_time)}")

        total_elapsed = time.time() - start_time
        print(f"[{get_timestamp()}] Generated {len(candidates)} candidates in {format_elapsed_time(total_elapsed)} (100% FREE!)")
        if len(candidates) > 0:
            avg_per_candidate = total_elapsed / len(candidates)
            print(f"[Timing] Average per candidate: {format_elapsed_time(avg_per_candidate)}")
        return parent, candidates

    except requests.exceptions.Timeout:
        print(f"[Ollama] ERROR: Request timeout (model may be loading)")
        print(f"[Ollama] Try running: ollama run {model_name} (to preload)")
        raise
    except Exception as e:
        print(f"[Ollama] ERROR: {e}")
        raise


def print_scenario_header(title: str, description: str):
    """Print formatted scenario header."""
    print("\n" + "=" * 80)
    print(f"SCENARIO: {title}")
    print("=" * 80)
    print(f"{description}\n")


def print_candidate_preview(candidate, index: int, show_score: bool = False, prefix: str = ""):
    """Print a single candidate with preview."""
    preview = candidate.text[:100].replace('\n', ' ')
    if len(candidate.text) > 100:
        preview += "..."

    if show_score and hasattr(candidate, 'score'):
        print(f"{prefix}{index}. [{candidate.id}] Score: {candidate.score:.4f}")
    else:
        print(f"{prefix}{index}. [{candidate.id}]")
    print(f"{prefix}   \"{preview}\"")


def print_detailed_results(result, scenario_name: str):
    """Print selection results with detailed stage-by-stage analysis."""

    print("\n" + "=" * 100)
    print(f"DETAILED PIPELINE ANALYSIS: {scenario_name}")
    print("=" * 100)

    # ========================================================================
    # STAGE 0: Initial Candidates (Before Scoring)
    # ========================================================================
    print("\n" + "-" * 100)
    print("STAGE 0: INITIAL CANDIDATES (Before Scoring)")
    print("-" * 100)
    print(f"Total candidates generated: {len(result.scored)}")
    print("\nAll candidates before any filtering:")
    for i, cand in enumerate(result.scored, 1):
        print_candidate_preview(cand, i, show_score=False, prefix="  ")

    # ========================================================================
    # STAGE 1: After Scoring
    # ========================================================================
    print("\n" + "-" * 100)
    print("STAGE 1: AFTER SCORING (Before Beam Selection)")
    print("-" * 100)
    print(f"All {len(result.scored)} candidates with their composite scores:")
    print("(Score = weighted combination of: confidence, relevance, novelty, intent_alignment)")
    print()

    # Sort by score to show ranking
    scored_sorted = sorted(result.scored, key=lambda c: c.score, reverse=True)
    for i, cand in enumerate(scored_sorted, 1):
        print_candidate_preview(cand, i, show_score=True, prefix="  ")

    # ========================================================================
    # STAGE 2: After Beam Selection (Top-K)
    # ========================================================================
    print("\n" + "-" * 100)
    print("STAGE 2: AFTER BEAM SELECTION (Top-K)")
    print("-" * 100)

    beam_kept_ids = {c.id for c in result.after_beam}
    beam_removed = [c for c in result.scored if c.id not in beam_kept_ids]

    print(f"Beam kept: {len(result.after_beam)} candidates (top-K)")
    print(f"Beam removed: {len(beam_removed)} candidates (low scores)")

    if result.after_beam:
        print(f"\n✅ KEPT by beam (top {len(result.after_beam)} by score):")
        for i, cand in enumerate(result.after_beam, 1):
            print_candidate_preview(cand, i, show_score=True, prefix="  ")

    if beam_removed:
        print(f"\n❌ REMOVED by beam (lower scores):")
        beam_removed_sorted = sorted(beam_removed, key=lambda c: c.score, reverse=True)
        for i, cand in enumerate(beam_removed_sorted, 1):
            print_candidate_preview(cand, i, show_score=True, prefix="  ")
            print(f"     Reason: Score {cand.score:.4f} below top-{len(result.after_beam)} threshold")

    # ========================================================================
    # STAGE 3: After Novelty Filtering
    # ========================================================================
    print("\n" + "-" * 100)
    print("STAGE 3: AFTER NOVELTY FILTERING (Diversity)")
    print("-" * 100)

    novelty_kept_ids = {c.id for c in result.after_novelty}
    novelty_removed = [c for c in result.after_beam if c.id not in novelty_kept_ids]

    print(f"Novelty kept: {len(result.after_novelty)} candidates (diverse)")
    print(f"Novelty removed: {len(novelty_removed)} candidates (too similar)")

    if result.after_novelty:
        print(f"\n✅ KEPT by novelty filter (diverse enough):")
        for i, cand in enumerate(result.after_novelty, 1):
            print_candidate_preview(cand, i, show_score=True, prefix="  ")

    if novelty_removed:
        print(f"\n❌ REMOVED by novelty filter (too similar to kept candidates):")
        for i, cand in enumerate(novelty_removed, 1):
            print_candidate_preview(cand, i, show_score=True, prefix="  ")
            print(f"     Reason: High similarity to already-selected candidates")

    # ========================================================================
    # STAGE 4: Entropy Decision
    # ========================================================================
    print("\n" + "-" * 100)
    print("STAGE 4: ENTROPY DECISION (Convergence Detection)")
    print("-" * 100)

    if result.entropy_decision:
        # Handle both dict and object cases
        if isinstance(result.entropy_decision, dict):
            entropy = result.entropy_decision.get('entropy', 0.0)
            should_continue = result.entropy_decision.get('should_continue', True)
            delta_entropy = result.entropy_decision.get('delta_entropy')
        else:
            entropy = result.entropy_decision.entropy
            should_continue = result.entropy_decision.should_continue
            delta_entropy = result.entropy_decision.delta_entropy

        print(f"Entropy (normalized): {entropy:.4f}")
        print(f"Should continue: {should_continue}")
        if delta_entropy is not None:
            print(f"Delta entropy: {delta_entropy:.4f}")

        # Explain the decision
        print("\nInterpretation:")
        if entropy >= 0.8:
            print("  → High entropy (≥0.8): Candidates are VERY diverse - continue exploring!")
        elif entropy >= 0.6:
            print("  → Medium-high entropy (0.6-0.8): Good diversity - keep exploring")
        elif entropy >= 0.4:
            print("  → Medium entropy (0.4-0.6): Moderate diversity - converging")
        elif entropy >= 0.2:
            print("  → Low entropy (0.2-0.4): Candidates becoming similar - near convergence")
        else:
            print("  → Very low entropy (<0.2): Candidates are very similar - STOP exploring")

        if should_continue:
            print("  → ✅ Decision: CONTINUE exploration (entropy still high enough)")
        else:
            print("  → 🛑 Decision: STOP exploration (convergence detected)")
    else:
        print("No entropy decision (entropy stopper not configured)")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 100)
    print("PIPELINE SUMMARY")
    print("=" * 100)

    print(f"\nFunnel visualization:")
    print(f"  Initial:        {len(result.scored):3d} candidates")
    print(f"  After Scoring:  {len(result.scored):3d} candidates (scored)")
    print(f"  After Beam:     {len(result.after_beam):3d} candidates (↓ {len(result.scored) - len(result.after_beam):2d} removed by beam)")
    print(f"  After Novelty:  {len(result.after_novelty):3d} candidates (↓ {len(result.after_beam) - len(result.after_novelty):2d} removed by novelty)")
    print(f"  Final Kept:     {len(result.kept):3d} candidates")

    # Reduction percentage
    if len(result.scored) > 0:
        reduction_pct = (1 - len(result.kept) / len(result.scored)) * 100
        print(f"\nTotal reduction: {reduction_pct:.1f}% ({len(result.scored)} → {len(result.kept)})")

    # Final kept candidates
    print(f"\n{'='*100}")
    print(f"FINAL {len(result.kept)} KEPT CANDIDATES (What you should use for next step)")
    print(f"{'='*100}")
    for i, cand in enumerate(result.kept, 1):
        print_candidate_preview(cand, i, show_score=True, prefix="  ")

    print("\n" + "=" * 100 + "\n")


def scenario_1_ai_memory(model_name: str = "qwen3:14b"):
    """
    Scenario 1: AI Memory Story

    Goal: Show entropy staying high across diverse genres.
    Demonstrates: Multiple creative approaches to the same prompt.
    """
    scenario_start = time.time()
    print_scenario_header(
        "1. AI Memory Story",
        "Goal: High temperature generation, diverse creative approaches.\n"
        "Expected: Entropy stays high - many different story genres/styles."
    )

    # Generate candidates
    gen_start = time.time()
    parent, candidates = generate_creative_candidates(
        prompt="Write three opening paragraphs for a short story about a lost AI regaining memory.",
        model_name=model_name,
        n=10,
        temperature=1.3,  # High temperature for creativity
        parent_id="ai-memory-parent"
    )
    gen_elapsed = time.time() - gen_start

    # Setup selector with FREE embeddings
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name="jinaai/jina-embeddings-v2-base-en"  # FREE, excellent quality
    )

    scorer = CompositeScorer(
        weights={
            "confidence": 0.2,
            "relevance": 0.3,
            "novelty_parent": 0.3,
            "intent_alignment": 0.2
        },
        embedding_provider=embedding_provider
    )

    beam = BeamSelector(scorer=scorer, k=7)

    # MMR for balancing quality and diversity
    novelty_filter = MMRNoveltyFilter(
        lambda_param=0.6,  # 60% relevance, 40% diversity
        threshold=0.80,
        embedding_provider=embedding_provider
    )

    entropy_stopper = ShannonEntropyStopper(
        min_entropy=0.7,  # Expect high entropy for diverse creative outputs
        k_max=5,
        embedding_provider=embedding_provider
    )

    budget_manager = BudgetManager(
        Budget(max_nodes=50, max_tokens=50000, max_ms=30000),
        mode="soft"
    )

    selector = BranchSelector(
        beam_selector=beam,
        novelty_filter=novelty_filter,
        entropy_stopper=entropy_stopper,
        budget_manager=budget_manager
    )

    # Run selection
    print(f"\n[{get_timestamp()}] Running selection pipeline...")
    pipeline_start = time.time()
    result = selector.step(parent, candidates)
    pipeline_elapsed = time.time() - pipeline_start

    print(f"[Timing] Pipeline execution: {format_elapsed_time(pipeline_elapsed)}")
    print_detailed_results(result, "AI Memory Story")

    scenario_elapsed = time.time() - scenario_start
    print(f"\n[Timing] Scenario 1 total time: {format_elapsed_time(scenario_elapsed)}")
    print(f"  - Generation: {format_elapsed_time(gen_elapsed)} ({gen_elapsed/scenario_elapsed*100:.1f}%)")
    print(f"  - Pipeline:   {format_elapsed_time(pipeline_elapsed)} ({pipeline_elapsed/scenario_elapsed*100:.1f}%)")
    print("\nINSIGHT: Notice the HIGH entropy - creative writing generates diverse ideas!")
    return result


def scenario_2_mars_detective(model_name: str = "qwen3:14b"):
    """
    Scenario 2: Mars Detective Plot Twists

    Goal: Demonstrate clustering of distinct idea clusters.
    Demonstrates: K-means clustering finds groups of similar plot ideas.
    """
    scenario_start = time.time()
    print_scenario_header(
        "2. Mars Detective Plot Twists",
        "Goal: Generate distinct plot twist ideas, show clustering.\n"
        "Expected: K-means finds clusters of similar twist types."
    )

    gen_start = time.time()
    parent, candidates = generate_creative_candidates(
        prompt="Suggest five plot twists for a detective story set on Mars.",
        model_name=model_name,
        n=12,
        temperature=1.4,  # Very high for wild creativity
        parent_id="mars-detective-parent"
    )
    gen_elapsed = time.time() - gen_start

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name="all-mpnet-base-v2"  # FREE, good quality, faster
    )

    scorer = CompositeScorer(
        weights={
            "confidence": 0.3,
            "relevance": 0.4,
            "novelty_parent": 0.3
        },
        embedding_provider=embedding_provider
    )

    beam = BeamSelector(scorer=scorer, k=8)

    # Cosine filter to remove very similar twists
    novelty_filter = CosineNoveltyFilter(
        threshold=0.75,  # Aggressive filtering for creativity
        embedding_provider=embedding_provider
    )

    entropy_stopper = ShannonEntropyStopper(
        min_entropy=0.65,
        k_max=4,  # Fewer clusters for distinct ideas
        embedding_provider=embedding_provider
    )

    budget_manager = BudgetManager(
        Budget(max_nodes=50, max_tokens=50000, max_ms=30000),
        mode="soft"
    )

    selector = BranchSelector(
        beam_selector=beam,
        novelty_filter=novelty_filter,
        entropy_stopper=entropy_stopper,
        budget_manager=budget_manager
    )

    print(f"\n[{get_timestamp()}] Running selection pipeline...")
    pipeline_start = time.time()
    result = selector.step(parent, candidates)
    pipeline_elapsed = time.time() - pipeline_start

    print(f"[Timing] Pipeline execution: {format_elapsed_time(pipeline_elapsed)}")
    print_detailed_results(result, "Mars Detective Twists")

    scenario_elapsed = time.time() - scenario_start
    print(f"\n[Timing] Scenario 2 total time: {format_elapsed_time(scenario_elapsed)}")
    print(f"  - Generation: {format_elapsed_time(gen_elapsed)} ({gen_elapsed/scenario_elapsed*100:.1f}%)")
    print(f"  - Pipeline:   {format_elapsed_time(pipeline_elapsed)} ({pipeline_elapsed/scenario_elapsed*100:.1f}%)")
    print("\nINSIGHT: K-means clustering groups similar plot twist types together!")
    return result


def scenario_3_romcom_endings(model_name: str = "qwen3:14b"):
    """
    Scenario 3: Rom-Com Multiple Endings

    Goal: Perfect demonstration of entropy-based stopping.
    Demonstrates: Clear stopping point when only 3 distinct ending types requested.
    """
    scenario_start = time.time()
    print_scenario_header(
        "3. Romantic Comedy Endings",
        "Goal: Generate exactly 3 ending types (tragic, absurd, heartwarming).\n"
        "Expected: Perfect entropy-based stopping - clear convergence to 3 clusters."
    )

    gen_start = time.time()
    parent, candidates = generate_creative_candidates(
        prompt="Describe three endings for the same romantic comedy: tragic, absurd, heartwarming.",
        model_name=model_name,
        n=15,
        temperature=1.2,
        parent_id="romcom-parent"
    )
    gen_elapsed = time.time() - gen_start

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name="BAAI/bge-large-en-v1.5"  # FREE, very good quality
    )

    scorer = CompositeScorer(
        weights={
            "confidence": 0.25,
            "relevance": 0.35,
            "novelty_parent": 0.25,
            "intent_alignment": 0.15
        },
        embedding_provider=embedding_provider
    )

    beam = BeamSelector(scorer=scorer, k=10)

    novelty_filter = MMRNoveltyFilter(
        lambda_param=0.5,  # Equal balance
        threshold=0.78,
        embedding_provider=embedding_provider
    )

    # Perfect for this scenario - exactly 3 clusters expected
    entropy_stopper = ShannonEntropyStopper(
        min_entropy=0.60,
        k_max=3,  # Exactly 3 ending types!
        embedding_provider=embedding_provider
    )

    budget_manager = BudgetManager(
        Budget(max_nodes=50, max_tokens=50000, max_ms=30000),
        mode="soft"
    )

    selector = BranchSelector(
        beam_selector=beam,
        novelty_filter=novelty_filter,
        entropy_stopper=entropy_stopper,
        budget_manager=budget_manager
    )

    print(f"\n[{get_timestamp()}] Running selection pipeline...")
    pipeline_start = time.time()
    result = selector.step(parent, candidates)
    pipeline_elapsed = time.time() - pipeline_start

    print(f"[Timing] Pipeline execution: {format_elapsed_time(pipeline_elapsed)}")
    print_detailed_results(result, "Rom-Com Endings")

    scenario_elapsed = time.time() - scenario_start
    print(f"\n[Timing] Scenario 3 total time: {format_elapsed_time(scenario_elapsed)}")
    print(f"  - Generation: {format_elapsed_time(gen_elapsed)} ({gen_elapsed/scenario_elapsed*100:.1f}%)")
    print(f"  - Pipeline:   {format_elapsed_time(pipeline_elapsed)} ({pipeline_elapsed/scenario_elapsed*100:.1f}%)")
    print("\nINSIGHT: With k_max=3, we expect 3 clusters (tragic, absurd, heartwarming)!")
    print("Perfect demonstration of entropy-based stopping for known categories.")
    return result


def scenario_4_style_variations(model_name: str = "qwen3:14b"):
    """
    Scenario 4: Style Variations (Hemingway, Austen, Murakami)

    Goal: Demonstrate intent alignment and style variance.
    Demonstrates: Detecting different writing styles as distinct intents.
    """
    scenario_start = time.time()
    print_scenario_header(
        "4. Style Variations",
        "Goal: Continue text in different literary styles.\n"
        "Expected: Intent alignment detects style variance, high novelty between styles."
    )

    # Custom prompt with a paragraph to continue
    starter_paragraph = (
        "The rain fell steadily on the cobblestone streets. "
        "She stood at the window, watching the droplets race down the glass."
    )

    prompt = (
        f"Continue this paragraph in three different styles:\n\n"
        f'"{starter_paragraph}"\n\n'
        f"1. In the style of Ernest Hemingway (terse, direct)\n"
        f"2. In the style of Jane Austen (elegant, ironic)\n"
        f"3. In the style of Haruki Murakami (surreal, dreamlike)"
    )

    gen_start = time.time()
    parent, candidates = generate_creative_candidates(
        prompt=prompt,
        model_name=model_name,
        n=12,
        temperature=1.1,
        parent_id="style-parent"
    )
    gen_elapsed = time.time() - gen_start

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name="jinaai/jina-embeddings-v2-base-en"  # FREE, excellent quality
    )

    scorer = CompositeScorer(
        weights={
            "confidence": 0.15,
            "relevance": 0.25,
            "novelty_parent": 0.35,  # High weight on novelty for style variance
            "intent_alignment": 0.25  # Important for detecting style adherence
        },
        embedding_provider=embedding_provider
    )

    beam = BeamSelector(scorer=scorer, k=9)

    # MMR to balance style similarity and distinctiveness
    novelty_filter = MMRNoveltyFilter(
        lambda_param=0.4,  # Favor diversity (60%) over relevance (40%)
        threshold=0.75,
        embedding_provider=embedding_provider
    )

    entropy_stopper = ShannonEntropyStopper(
        min_entropy=0.65,
        k_max=3,  # 3 distinct styles
        embedding_provider=embedding_provider
    )

    budget_manager = BudgetManager(
        Budget(max_nodes=50, max_tokens=50000, max_ms=30000),
        mode="soft"
    )

    selector = BranchSelector(
        beam_selector=beam,
        novelty_filter=novelty_filter,
        entropy_stopper=entropy_stopper,
        budget_manager=budget_manager
    )

    print(f"\n[{get_timestamp()}] Running selection pipeline...")
    pipeline_start = time.time()
    result = selector.step(parent, candidates)
    pipeline_elapsed = time.time() - pipeline_start

    print(f"[Timing] Pipeline execution: {format_elapsed_time(pipeline_elapsed)}")
    print_detailed_results(result, "Style Variations")

    scenario_elapsed = time.time() - scenario_start
    print(f"\n[Timing] Scenario 4 total time: {format_elapsed_time(scenario_elapsed)}")
    print(f"  - Generation: {format_elapsed_time(gen_elapsed)} ({gen_elapsed/scenario_elapsed*100:.1f}%)")
    print(f"  - Pipeline:   {format_elapsed_time(pipeline_elapsed)} ({pipeline_elapsed/scenario_elapsed*100:.1f}%)")
    print("\nINSIGHT: High novelty scores between different literary styles!")
    print("Intent alignment helps ensure each style is distinct.")
    return result


def main():
    """Run all creative writing scenarios."""
    print("=" * 80)
    print("ChatRoutes AutoBranch - Creative Writing (100% FREE!)")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  - Ollama models (qwen3, gpt-oss, or llama3) for text generation (100% FREE!)")
    print("  - FREE sentence-transformers embeddings for comparison (NO COST!)")
    print("  - High temperature (1.1-1.4) for creative diversity")
    print("  - Entropy-based stopping for convergence detection")
    print("  - MMR and Cosine novelty filtering")
    print("\n100% FREE Strategy:")
    print("  - Chat generation: Ollama (auto-detects best available model)")
    print("  - Embeddings: sentence-transformers (runs locally, $0)")
    print("  - Total cost: $0 - Everything runs on your machine!")
    print("\nREQUIREMENTS:")
    print("  - pip install sentence-transformers requests")
    print("  - Install Ollama from https://ollama.ai")
    print("  - Install ONE of these models:")
    print("    • RECOMMENDED: ollama pull qwen3:14b (fast, excellent quality)")
    print("    • ALTERNATIVE: ollama pull gpt-oss:20b (OpenAI model, slower)")
    print("    • FALLBACK:    ollama pull llama3.1:8b (fast, good baseline)")
    print("  - First run will download embedding models (~400MB each)")

    # Check Ollama availability and determine model
    print("\nChecking Ollama availability...")
    is_available, model_name = check_ollama_available()

    if not is_available:
        print("\nPlease install Ollama and download a model:")
        print("  1. Download Ollama from https://ollama.ai")
        print("  2. Install and start Ollama (auto-starts on Windows)")
        print("  3. RECOMMENDED: ollama pull qwen3:14b")
        print("  4. FALLBACK: ollama pull llama3.1:8b")
        print("  5. Test: ollama run qwen3:14b")
        return

    print(f"[OK] Using model: {model_name}")
    print(f"\n[{get_timestamp()}] Starting 4 creative writing scenarios...\n")

    # Run all scenarios
    overall_start = time.time()
    try:
        # Scenario 1: AI Memory Story
        print(f"[{get_timestamp()}] Starting Scenario 1...")
        scenario_1_ai_memory(model_name=model_name)

        # Scenario 2: Mars Detective Twists
        print(f"\n[{get_timestamp()}] Starting Scenario 2...")
        scenario_2_mars_detective(model_name=model_name)

        # Scenario 3: Rom-Com Endings
        print(f"\n[{get_timestamp()}] Starting Scenario 3...")
        scenario_3_romcom_endings(model_name=model_name)

        # Scenario 4: Style Variations
        print(f"\n[{get_timestamp()}] Starting Scenario 4...")
        scenario_4_style_variations(model_name=model_name)

        # Summary
        overall_elapsed = time.time() - overall_start
        print("\n" + "=" * 80)
        print(f"ALL SCENARIOS COMPLETE - Total Time: {format_elapsed_time(overall_elapsed)}")
        print("=" * 80)

        print("\n[TIMING SUMMARY]")
        print(f"  Total execution time: {format_elapsed_time(overall_elapsed)}")
        print(f"  Average per scenario: {format_elapsed_time(overall_elapsed / 4)}")

        print("\nKey Takeaways:")
        print("  1. High temperature → High diversity → High entropy")
        print("  2. K-means clustering groups similar creative ideas")
        print("  3. Entropy-based stopping works well for known categories")
        print("  4. MMR balances quality and diversity effectively")
        print("  5. FREE embeddings work great for semantic similarity!")
        print("  6. 100% FREE approach - runs completely on your machine!")
        print("\nCreative writing is a perfect use case for chatroutes-autobranch!")
        print("\nCost Summary:")
        print(f"  - Chat generation: $0 (Ollama {model_name}, runs locally)")
        print("  - Embeddings: $0 (sentence-transformers, runs locally)")
        print("  - Total cost: $0 - Completely free and private!")

        if model_name == "llama3.1:8b":
            print("\nTIP: For better creative writing quality, try:")
            print("  ollama pull qwen3:14b")
        elif model_name == "gpt-oss:20b":
            print("\nNOTE: gpt-oss:20b is slow but high quality.")
            print("For faster execution (20x), consider:")
            print("  ollama pull qwen3:14b")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nCommon issues:")
        print("  - Ollama not running (start Ollama)")
        print("  - Model not downloaded:")
        print("    • ollama pull qwen3:14b (recommended)")
        print("    • ollama pull llama3.1:8b (fallback)")
        print("  - Insufficient RAM:")
        print("    • qwen3:14b needs ~16GB")
        print("    • llama3.1:8b needs ~8GB")
        print("  - Network connectivity (for first-time model download)")


if __name__ == "__main__":
    main()
