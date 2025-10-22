# chatroutes-autobranch

**Controlled branching generation for LLM applications**

[![PyPI version](https://badge.fury.io/py/chatroutes-autobranch.svg)](https://badge.fury.io/py/chatroutes-autobranch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chatroutes/chatroutes-autobranch/blob/master/notebooks/creative_writing_colab.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modern LLM applications often need to explore multiple reasoning paths (tree-of-thought, beam search, multi-agent systems) while staying **usable** and **affordable**. `chatroutes-autobranch` provides clean, standalone primitives for:

- 🎯 **Beam Search** – Pick the *best K* candidates by configurable scoring
- 🌈 **Diversity Control** – Ensure variety via novelty pruning (cosine similarity, MMR)
- 🛑 **Smart Stopping** – Know when to stop via entropy/information-gain metrics
- 💰 **Budget Management** – Keep costs predictable with token/time/node caps
- 🔌 **Pluggable Design** – Swap any component (scorer, embeddings, stopping criteria)

**Key Features:**
- ✅ Deterministic & reproducible (fixed tie-breaking, seeded clustering)
- ✅ Embedding-agnostic (OpenAI, HuggingFace, or custom)
- ✅ Production-ready (thread-safe, observable, checkpoint/resume)
- ✅ Framework-friendly (works with LangChain, LlamaIndex, or raw LLM APIs)
- ✅ Zero vendor lock-in (MIT License, no cloud dependencies)

---

## 🚀 Interactive Demos (Try it Now!)

### Getting Started Demo (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chatroutes/chatroutes-autobranch/blob/master/notebooks/getting_started_demo.ipynb)

**Perfect for first-time users!** Learn the fundamentals in 5 minutes:
- ✅ Installation and setup
- ✅ Basic beam search examples
- ✅ Multi-strategy scoring
- ✅ Novelty filtering
- ✅ Complete pipeline with budget control

**No setup required** - runs entirely in your browser!

### Creative Writing Scenario (Advanced)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chatroutes/chatroutes-autobranch/blob/master/notebooks/creative_writing_colab.ipynb)

**See it in action with a real LLM!** Complete creative writing assistant:
- ✅ Full Ollama integration (free, local inference)
- ✅ Multi-turn branching (tree exploration)
- ✅ GPU/CPU performance comparison
- ✅ 4 complete story scenarios

[**📚 View all notebooks →**](notebooks/README.md)

---

## Quick Start

**Install:**
```bash
pip install chatroutes-autobranch
```

**Basic Usage:**
```python
from chatroutes_autobranch import BranchSelector, Candidate
from chatroutes_autobranch.config import load_config

# Load config (or use dict/env vars)
selector = BranchSelector.from_config(load_config("config.yaml"))

# Define parent and candidate branches
parent = Candidate(id="root", text="Explain photosynthesis simply")
candidates = [
    Candidate(id="c1", text="Start with sunlight absorption"),
    Candidate(id="c2", text="Begin with glucose production"),
    Candidate(id="c3", text="Explain chlorophyll's role"),
]

# Select best branches (applies beam → novelty → entropy pipeline)
result = selector.step(parent, candidates)

print(f"Kept: {[c.id for c in result.kept]}")
print(f"Entropy: {result.metrics['entropy']['value']:.2f}")
print(f"Should continue: {result.metrics['entropy']['continue']}")
```

**Config (`config.yaml`):**
```yaml
beam:
  k: 3  # Keep top 3 by score
  weights: {confidence: 0.4, relevance: 0.3, novelty_parent: 0.2}

novelty:
  method: cosine  # or 'mmr' for Maximal Marginal Relevance
  threshold: 0.85

entropy:
  min_entropy: 0.6  # Stop if diversity drops below 60%

embeddings:
  provider: openai
  model: text-embedding-3-large
```

---

## Why Use This?

**Problem:** Exploring multiple LLM reasoning paths (e.g., tree-of-thought) quickly becomes:
- **Expensive** – Exponential growth of branches drains API budgets
- **Redundant** – Models generate similar outputs (mode collapse)
- **Uncontrolled** – No clear stopping criteria (when is "enough" exploration?)

**Solution:** `chatroutes-autobranch` gives you:
1. **Beam Search** to keep only the top-K candidates (quality filtering)
2. **Novelty Pruning** to remove similar outputs (diversity enforcement)
3. **Entropy Stopping** to detect when you've explored enough (convergence detection)
4. **Budget Limits** to cap costs before runaway spending

**Result:** Controlled, efficient tree exploration with predictable costs.

---

## Use Cases

| Scenario | Configuration | Benefit |
|----------|--------------|---------|
| **Tree-of-Thought Reasoning** | K=5, cosine novelty, entropy stopping | Explore diverse reasoning paths without explosion |
| **Multi-Agent Debate** | K=3, MMR novelty (λ=0.3) | Select diverse agent perspectives, avoid redundancy |
| **Code Generation** | K=4, high relevance weight | Generate varied solutions, prune duplicates |
| **Creative Writing** | K=8, low novelty threshold | High diversity, explore creative space |
| **Factual Q&A** | K=2, strict budget | Focus on accuracy, minimal branching |

---

## Architecture

**Pipeline (fixed order):**
```
Raw Candidates (N)
    ↓
1. Scoring (composite: confidence + relevance + novelty + intent + reward)
    ↓
2. Beam Selection (top K by score, deterministic tie-breaking)
    ↓
3. Novelty Filtering (prune similar via cosine/MMR)
    ↓
4. Entropy Check (compute diversity, decide if should continue)
    ↓
5. Result (kept + pruned + metrics)
```

**Pluggable Components:**
- **Scorer**: Composite (built-in) or custom
- **EmbeddingProvider**: OpenAI, HuggingFace, or custom
- **NoveltyFilter**: Cosine threshold or MMR
- **EntropyStopper**: Shannon entropy or custom
- **BudgetManager**: Token/time/node caps

All components use **Protocol** (duck typing) – swap any part without touching others.

---

## Installation

**Minimal:**
```bash
pip install chatroutes-autobranch
```

**With extras:**
```bash
# FastAPI service (for TypeScript/other languages)
pip install chatroutes-autobranch[service]

# HuggingFace local embeddings
pip install chatroutes-autobranch[hf]

# FAISS for large-scale similarity (1000+ candidates)
pip install chatroutes-autobranch[faiss]

# All features
pip install chatroutes-autobranch[all]
```

---

## Documentation

📘 **[Full Specification](./chatroutes_autobranch_v1.0.md)** – Complete API reference, algorithms, examples, and troubleshooting

**Key Sections:**
- [Philosophy & Design](./chatroutes_autobranch_v1.0.md#0-philosophy) – Core principles
- [Pluggable Interfaces](./chatroutes_autobranch_v1.0.md#4-variability-pluggable-interfaces) – Protocols & implementations
- [Configuration](./chatroutes_autobranch_v1.0.md#5-configuration) – YAML/JSON/env setup
- [Examples](./chatroutes_autobranch_v1.0.md#6-example-usage) – Single-step & multi-generation
- [Tuning Guide](./chatroutes_autobranch_v1.0.md#19-tuning-guide-choosing-beam-width-k) – How to choose K
- [Common Failures](./chatroutes_autobranch_v1.0.md#20-common-failure-patterns) – Troubleshooting

---

## Examples

### Multi-Generation Tree Exploration

```python
from collections import deque
import time

# User provides LLM generation function
def my_llm_generate(parent: Candidate, n: int) -> list[Candidate]:
    # Your LLM call here (OpenAI, Anthropic, etc.)
    responses = llm_api.generate(parent.text, n=n)
    return [Candidate(id=f"{parent.id}_{i}", text=r) for i, r in enumerate(responses)]

# Setup
selector = BranchSelector.from_config(load_config("config.yaml"))
budget_manager = BudgetManager(Budget(max_nodes=50, max_tokens=20000))

# Tree exploration
queue = deque([root_candidate])
while queue:
    current = queue.popleft()
    children = my_llm_generate(current, n=5)

    # Check budget before selection
    if not budget_manager.admit(n_new=5, est_tokens=1000, est_ms=2000):
        break

    # Select best branches
    result = selector.step(current, children)
    budget_manager.update(actual_tokens=1200, actual_ms=1800)

    # Continue with kept candidates
    queue.extend(result.kept)

    # Stop if entropy is low (converged)
    if not result.metrics["entropy"]["continue"]:
        break
```

### Custom Scorer

```python
from chatroutes_autobranch import Scorer, Candidate, ScoredCandidate

class DomainScorer(Scorer):
    def score(self, parent: Candidate, candidates: list[Candidate]) -> list[ScoredCandidate]:
        scored = []
        for c in candidates:
            # Custom logic: prefer longer, detailed responses
            detail_score = min(len(c.text) / 1000, 1.0)
            scored.append(ScoredCandidate(id=c.id, text=c.text, score=detail_score))
        return scored

# Use in pipeline
beam = BeamSelector(k=3, scorer=DomainScorer())
selector = BranchSelector(beam, novelty, entropy, budget)
```

### FastAPI Service (for TypeScript/other languages)

```python
# server.py
from fastapi import FastAPI
from chatroutes_autobranch import BranchSelector
from chatroutes_autobranch.config import load_config_from_file

app = FastAPI()
_config = load_config_from_file("config.yaml")

@app.post("/select")
async def select(parent: dict, candidates: list[dict]):
    # Create fresh selector per request (thread-safe)
    selector = BranchSelector.from_config(_config)
    result = selector.step(
        Candidate(**parent),
        [Candidate(**c) for c in candidates]
    )
    return {
        "kept": [{"id": c.id, "score": c.score} for c in result.kept],
        "metrics": result.metrics
    }

# Run: uvicorn server:app
```

**TypeScript client:**
```typescript
const response = await fetch('http://localhost:8000/select', {
  method: 'POST',
  body: JSON.stringify({ parent, candidates })
});
const { kept, metrics } = await response.json();
```

---

## Features

### Beam Search
- Top-K selection by composite scoring
- Deterministic tie-breaking (lexicographic ID ordering)
- Configurable weights: confidence, relevance, novelty, intent alignment, historical reward

### Novelty Pruning
- **Cosine similarity:** Remove candidates above threshold (e.g., 0.85)
- **MMR (Maximal Marginal Relevance):** Balance relevance vs diversity with λ parameter
- Preserves score ordering (best candidates kept first)

### Entropy-Based Stopping
- Shannon entropy on K-means clusters of embeddings
- Delta-entropy tracking (stop if change < epsilon)
- Handles edge cases (0, 1, 2 candidates)
- Normalized to [0,1] scale

### Budget Management
- **Caps:** max_nodes, max_tokens, max_ms
- **Modes:** strict (raise on exceeded) or soft (return False, allow fallback)
- **Pre-admit:** Check budget before generation
- **Post-update:** Record actual usage for rolling averages

### Observability
- Structured JSON logging (PII-safe by default)
- OpenTelemetry spans (optional)
- Rich metrics per step (kept/pruned counts, scores, entropy, budget usage)

### Checkpointing
- Serialize selector state (entropy history, budget snapshot)
- Resume from checkpoint (pause/resume tree exploration)
- Schema versioning for backward compatibility

---

## Integrations

**LangChain:**
```python
from langchain.chains import LLMChain
from chatroutes_autobranch import Candidate, BranchSelector

def generate_and_select(query: str, chain: LLMChain, selector: BranchSelector):
    # Generate N candidates via LangChain
    responses = chain.generate([{"query": query}] * 5)
    candidates = [Candidate(id=f"c{i}", text=r.text) for i, r in enumerate(responses.generations[0])]

    # Select best
    parent = Candidate(id="root", text=query)
    result = selector.step(parent, candidates)
    return result.kept
```

**LlamaIndex:** Similar pattern using `QueryEngine.query()` for generation

**Raw APIs (OpenAI, Anthropic):** See [multi-generation example](#multi-generation-tree-exploration)

---

## Performance

**Benchmarks** (M1 Max, OpenAI embeddings):

| Candidates | Beam K | Latency (p50) | Bottleneck |
|-----------|--------|---------------|------------|
| 10 | 3 | 240ms | Embedding API |
| 50 | 5 | 520ms | Embedding API |
| 100 | 10 | 1.1s | Novelty O(N²) |
| 500 | 10 | 4.2s | Use FAISS |

**Optimization tips:**
- Use local embeddings (HuggingFace) for <100ms latency
- Enable FAISS for 100+ candidates
- Batch embedding calls (`batch_size: 64` in config)
- Global embedding cache for repeated candidates

---

## Development

**Setup:**
```bash
git clone https://github.com/chatroutes/chatroutes-autobranch
cd chatroutes-autobranch
pip install -e .[dev]
```

**Run tests:**
```bash
pytest tests/
pytest tests/ -v --cov=chatroutes_autobranch  # With coverage
```

**Type checking:**
```bash
mypy src/
```

**Formatting:**
```bash
black src/ tests/
ruff check src/ tests/
```

**Benchmarks:**
```bash
pytest bench/ --benchmark-only
```

---

## Contributing

We welcome contributions! Please see our [contributing guidelines](./CONTRIBUTING.md).

**Areas we'd love help with:**
- Additional novelty algorithms (DPP, k-DPP)
- More embedding providers (Cohere, Voyage AI)
- Adaptive K scheduling (auto-tune beam width)
- Tree visualization tools
- More examples (specific domains)

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run tests and type checking
5. Submit a Pull Request

---

## Roadmap

- **v1.0.0** ✅ **RELEASED** (January 2025): Core components, beam search, MMR novelty, cosine filtering, entropy stopping, budget management, full test suite
- **v1.1.0** (Q2 2025): FAISS support for large-scale similarity, adaptive K scheduling
- **v1.2.0** (Q3 2025): Tree visualization tools, FastAPI service for multi-language support
- **v1.3.0** (Q4 2025): Async/await support, cluster-aware pruning
- **v2.0.0** (Q1 2026): gRPC service, TypeScript SDK, breaking API improvements

---

## FAQ

**Q: Do I need ChatRoutes cloud to use this?**
A: No. This library is standalone and has zero cloud dependencies. Use it with any LLM provider.

**Q: Can I use this with TypeScript/JavaScript?**
A: Yes. Run the FastAPI service and call via HTTP. Native TS SDK planned for v2.0.0.

**Q: How do I choose beam width K?**
A: Start with K=3-5. Use budget formula: `K ≈ (budget/tokens_per_branch)^(1/depth)`. See [tuning guide](./chatroutes_autobranch_v1.0.md#19-tuning-guide-choosing-beam-width-k).

**Q: What if all candidates get pruned by novelty?**
A: Lower threshold (e.g., 0.75) or switch to MMR. See [troubleshooting](./chatroutes_autobranch_v1.0.md#20-common-failure-patterns).

**Q: Is this deterministic?**
A: Yes, with fixed random seeds and deterministic tie-breaking. See [tests](./chatroutes_autobranch_v1.0.md#7-tests-pytest).

---

## License

MIT License - see [LICENSE](./LICENSE) file for details.

---

## Acknowledgements

Inspired by research in beam search, diverse selection (MMR, DPP), and LLM orchestration patterns. Built to be practical, swappable, and friendly for contributors.

Special thanks to the open-source community for tools and inspiration: LangChain, LlamaIndex, HuggingFace Transformers, FAISS, and the broader LLM ecosystem.

---

## Links

- **Documentation:** [Full Specification](./chatroutes_autobranch_v1.0.md)
- **Issues:** [GitHub Issues](https://github.com/chatroutes/chatroutes-autobranch/issues)
- **Discussions:** [GitHub Discussions](https://github.com/chatroutes/chatroutes-autobranch/discussions)
- **Changelog:** [CHANGELOG.md](./CHANGELOG.md)
- **PyPI:** [pypi.org/project/chatroutes-autobranch](https://pypi.org/project/chatroutes-autobranch)

---

**Built with ❤️ by the ChatRoutes team. Open to the community.**
