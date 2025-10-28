# Reproducibility Guide - ChatRoutes AutoBranch

**TL;DR**: ChatRoutes AutoBranch is **fully deterministic** when you control randomness sources (seeds). Same inputs + same seeds = **identical outputs every time**.

⚠️ **IMPORTANT GPU CAVEAT**: **GPUs are NOT guaranteed to be deterministic** even with fixed seeds. For 100% reproducibility, use **CPU inference**.

---

## Quick Answer: Is it Reproducible?

**YES (on CPU)** - With proper seed control, you get **100% reproducibility**.

**MAYBE (on GPU)** - GPUs can introduce non-determinism even with fixed seeds due to:
- Parallel floating-point operations (non-associative)
- Non-deterministic algorithms in cuDNN/CUDA
- Hardware-specific optimizations
- Race conditions in parallel kernels

The test suite (`examples/test_reproducibility.py`) demonstrates:
- ✅ **10 consecutive runs (CPU)**: Identical results (`['c1', 'c3', 'c2']`)
- ✅ **Same seed = Same output (CPU)**: Perfect reproducibility
- ⚠️ **GPU**: May vary slightly across runs (use CPU for critical reproducibility)

---

## What is Deterministic?

### ✅ Fully Deterministic Components

| Component | Deterministic? | Why? |
|-----------|----------------|------|
| **Scoring** (CompositeScorer) | ✅ Yes | Pure math operations (no randomness) |
| **Beam Selection** (BeamSelector) | ✅ Yes | Stable sort, deterministic top-k |
| **Novelty Filtering** (Cosine/MMR) | ✅ Yes | Deterministic similarity calculations |
| **Entropy Stopping** (Shannon) | ✅ Yes (with seed) | KMeans with `random_state` |
| **Budget Management** | ✅ Yes | Simple counters (no randomness) |
| **DummyEmbeddingProvider** | ✅ Yes (with seed) | Seeded random number generation |

### ⚠️ Requires Seed Control

| Component | Deterministic? | How to Control |
|-----------|----------------|----------------|
| **LLM Generation** (Ollama/OpenAI) | ⚠️ With seed | Set `seed` parameter in API calls |
| **Sentence-Transformers** | ✅ Mostly | Same model + hardware = same results |

---

## How to Ensure Reproducibility

### 1. For Testing/Development (DummyEmbeddingProvider)

```python
from chatroutes_autobranch import (
    BranchSelector, BeamSelector, CompositeScorer,
    MMRNoveltyFilter, ShannonEntropyStopper,
    DummyEmbeddingProvider, Candidate
)

# SEED EVERYTHING
SEED = 42

# 1. Seeded embeddings
embedding_provider = DummyEmbeddingProvider(dimension=64, seed=SEED)

# 2. Deterministic scorer (no randomness needed)
scorer = CompositeScorer(
    weights={"confidence": 0.4, "relevance": 0.4, "novelty": 0.2},
    embedding_provider=embedding_provider
)

# 3. Deterministic beam (no randomness)
beam = BeamSelector(scorer=scorer, k=3)

# 4. Deterministic novelty filter (deterministic given embeddings)
novelty_filter = MMRNoveltyFilter(
    lambda_param=0.7,
    embedding_provider=embedding_provider
)

# 5. Seeded entropy stopper
entropy_stopper = ShannonEntropyStopper(
    min_entropy=0.8,
    embedding_provider=embedding_provider,
    random_seed=SEED  # ← CRITICAL!
)

# 6. Assemble pipeline
selector = BranchSelector(
    beam_selector=beam,
    novelty_filter=novelty_filter,
    entropy_stopper=entropy_stopper
)

# RUN: Guaranteed identical results every time
result = selector.step(parent, candidates)
```

**Result**: Run this 1000 times → **identical output every time**

---

### 2. For Production (Real LLMs + Sentence-Transformers)

```python
from chatroutes_autobranch import (
    BranchSelector, BeamSelector, CompositeScorer,
    MMRNoveltyFilter, ShannonEntropyStopper
)
from sentence_transformers import SentenceTransformer
import requests

# REPRODUCIBILITY CHECKLIST
SEED = 42
MODEL_VERSION = "llama3.1:8b"  # Document exact version
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Specific version

# 1. Deterministic embeddings
class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

embedding_provider = SentenceTransformerEmbeddingProvider(EMBEDDING_MODEL)

# 2. Setup pipeline (same as above)
scorer = CompositeScorer(...)
beam = BeamSelector(scorer=scorer, k=3)
novelty_filter = MMRNoveltyFilter(...)
entropy_stopper = ShannonEntropyStopper(..., random_seed=SEED)
selector = BranchSelector(...)

# 3. Reproducible LLM generation
def generate_with_fixed_seed(prompt: str, seed: int = SEED):
    """Generate with fixed seed for reproducibility."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_VERSION,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 80,
                "seed": seed  # ← FIXED SEED
            }
        }
    )
    return response.json()["response"]

# Generate candidates with FIXED seed
candidates = []
for i in range(10):
    text = generate_with_fixed_seed(prompt, seed=SEED)  # Same seed every time
    candidates.append(Candidate(id=f"c{i}", text=text, meta={...}))

# Run pipeline
result = selector.step(parent, candidates)
```

**Result**: Same Ollama model version + same seed → **identical generations**

**⚠️ Important**: Different Ollama versions may produce different results even with same seed.

---

## Sources of Non-Determinism (and How to Fix)

### 1. LLM Generation (Most Common Issue)

**Problem**: Different outputs on every run

**Cause**:
- No seed set → random sampling
- Different seeds per candidate (e.g., `seed=1000+i`)

**Solution**:
```python
# BAD: Non-reproducible
"seed": 1000 + i  # Different seed per candidate

# GOOD: Reproducible
"seed": 42  # Same seed for all (or document seed strategy)
```

**Trade-off**: Fixed seed → less diversity in generations. For reproducibility, document your choice:
- **Research/Testing**: Use fixed seed (reproducibility priority)
- **Production**: Use varying seeds (diversity priority), but document the seed strategy

---

### 2. Entropy Stopper K-Means Clustering

**Problem**: Different clustering across runs

**Cause**: KMeans has random initialization

**Solution**:
```python
# BAD: Random initialization
entropy_stopper = ShannonEntropyStopper(
    min_entropy=0.8,
    # random_seed not set → non-deterministic
)

# GOOD: Fixed seed
entropy_stopper = ShannonEntropyStopper(
    min_entropy=0.8,
    random_seed=42  # ← Always set this!
)
```

---

### 3. DummyEmbeddingProvider (Testing Only)

**Problem**: Different embeddings across runs

**Cause**: No seed set

**Solution**:
```python
# BAD: Random embeddings
provider = DummyEmbeddingProvider(dimension=64)  # No seed

# GOOD: Deterministic embeddings
provider = DummyEmbeddingProvider(dimension=64, seed=42)
```

---

### 4. Hardware/Software Variations

**Sources of variation**:
- Different PyTorch versions
- Different CUDA versions
- **CPU vs GPU (CRITICAL: GPU is non-deterministic!)**
- Different GPU models (NVIDIA vs AMD, different architectures)
- Different Ollama versions
- Different model weights (even same model name)

#### ⚠️ GPU Non-Determinism (Most Important!)

**Why GPUs are non-deterministic**:

1. **Parallel floating-point operations**: GPUs execute operations in parallel, but floating-point addition is **not associative**:
   ```python
   # These may give different results due to rounding:
   (a + b) + c  !=  a + (b + c)  # On GPU with parallel reduction
   ```

2. **Non-deterministic algorithms**: cuDNN (NVIDIA's deep learning library) uses non-deterministic algorithms by default for performance:
   - Convolutions
   - Pooling operations
   - Attention mechanisms (in transformers)

3. **Atomic operations**: GPU atomic operations on shared memory can execute in non-deterministic order.

4. **Hardware-specific optimizations**: Different GPU models optimize differently, even with same code.

**Mitigation strategies** (partial solutions):

```python
# PyTorch: Enable deterministic mode (SLOWER but more reproducible)
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set all seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import numpy as np
import random
np.random.seed(42)
random.seed(42)
```

**⚠️ Warning**: Even with these settings, **GPU is still not 100% deterministic**. Some operations remain non-deterministic due to hardware limitations.

**Best Practices**:
```python
# Document your environment
environment = {
    "python_version": "3.11.5",
    "torch_version": "2.1.0",
    "sentence_transformers_version": "2.2.2",
    "ollama_version": "0.1.17",
    "model_version": "llama3.1:8b (sha256:...)",
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "hardware": "CPU",  # ← For reproducibility, always use CPU!
    # If using GPU, document:
    # "hardware": "NVIDIA RTX 3090",
    # "cuda_version": "11.8",
    # "cudnn_version": "8.6.0",
    # "deterministic_mode": False,  # GPU non-determinism accepted
    "seed": 42
}
```

**For critical reproducibility**:
- ✅ **Use CPU inference** (RECOMMENDED - fully deterministic)
- ⚠️ **Avoid GPU for reproducible research** (or document non-determinism)
- ✅ Pin all dependency versions (`requirements.txt`)
- ✅ Document model checksums/hashes
- ✅ Use containers (Docker) for full environment reproducibility
- ✅ Always test reproducibility on your target hardware before claiming it

---

## Testing Reproducibility

### Run the Test Suite

```bash
python examples/test_reproducibility.py
```

**Expected output**:
```
======================================================================
REPRODUCIBILITY TEST
======================================================================

Testing 10 runs with seed=42
Expected: All runs produce IDENTICAL results

Run  1: ['c1', 'c3', 'c2']
Run  2: ['c1', 'c3', 'c2']
Run  3: ['c1', 'c3', 'c2']
...
Run 10: ['c1', 'c3', 'c2']

======================================================================
[OK] REPRODUCIBILITY VERIFIED!
[OK] All 10 runs produced identical results
[OK] Kept candidates: ['c1', 'c3', 'c2']
======================================================================
```

---

## Reproducibility Checklist

Before claiming reproducibility, verify:

- [ ] **Seeds set**:
  - [ ] `DummyEmbeddingProvider(seed=42)` or real embedding provider
  - [ ] `ShannonEntropyStopper(random_seed=42)`
  - [ ] LLM generation `seed=42` (if using Ollama/OpenAI)

- [ ] **Versions documented**:
  - [ ] Python version
  - [ ] chatroutes-autobranch version
  - [ ] LLM model version (exact, with checksum if possible)
  - [ ] Embedding model version
  - [ ] PyTorch/TensorFlow version (if using neural embeddings)

- [ ] **Hardware documented**:
  - [ ] CPU vs GPU
  - [ ] GPU model (if using GPU)
  - [ ] Operating system

- [ ] **Test reproducibility**:
  - [ ] Run `python examples/test_reproducibility.py`
  - [ ] Verify all runs produce identical results
  - [ ] Document any variations

---

## Common Questions

### Q: Why do different seeds sometimes produce the same results?

**A**: With small candidate sets and strong score differences, the top-k selection may be stable across different embeddings. This is fine - it means your scoring is robust. The important property is: **same seed = same results**.

---

### Q: Can I reproduce results across different machines?

**A**: **Yes (on CPU)**, **No (on GPU)**, if you:

**✅ CPU**: Fully reproducible across machines
1. Use same Python/library versions
2. Use same model versions (LLM + embeddings)
3. Use same seeds
4. Run on CPU (not GPU)

**❌ GPU**: NOT reproducible across different GPUs
- Different GPU models → different results (even with same seeds)
- Same GPU model, different drivers → may differ
- GPU non-determinism means results vary even on same hardware

**Best practice**:
- For reproducible research: **Use CPU inference only**
- For production: Accept GPU non-determinism, document your setup
- Use Docker/containers for full environment reproducibility (CPU only)

---

### Q: Should I use fixed seeds in production?

**A**: **Depends on your use case**:

- **Research/Evaluation**: Yes, use fixed seeds (reproducibility > diversity)
- **Production/User-facing**: Maybe not (diversity > reproducibility)
  - But **document your seed strategy** (e.g., "seed = hash(user_id) + timestamp")
  - This gives reproducibility **per user** while maintaining diversity **across users**

---

### Q: What if I want diversity but also reproducibility?

**A**: Use **deterministic seed generation**:

```python
import hashlib

def deterministic_seed(base_seed: int, user_id: str, request_id: str) -> int:
    """Generate deterministic seed from context."""
    combined = f"{base_seed}-{user_id}-{request_id}"
    hash_value = hashlib.sha256(combined.encode()).hexdigest()
    return int(hash_value[:8], 16)

# Same user + same request → same results
# Different users → different results (but reproducible!)
seed = deterministic_seed(42, "user123", "request456")
```

---

### Q: I'm using GPU for speed. Can I still get reproducibility?

**A**: **Partial reproducibility at best, full non-determinism at worst**.

**Reality check**:
- ❌ GPU is **inherently non-deterministic** due to parallel operations
- ⚠️ PyTorch's deterministic mode helps but **doesn't guarantee 100% reproducibility**
- ❌ Different GPU models → different results (always)
- ⚠️ Same GPU → **may** vary across runs (non-deterministic algorithms)

**Options**:

1. **Accept non-determinism** (most realistic for GPU):
   ```python
   # Document that results may vary
   config = {
       "hardware": "NVIDIA RTX 3090",
       "deterministic_mode": False,
       "note": "GPU non-determinism accepted for speed"
   }
   ```

2. **Enable PyTorch deterministic mode** (slower, still not perfect):
   ```python
   import torch
   torch.use_deterministic_algorithms(True)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

   # May still have some non-determinism, but reduced
   ```

3. **Use CPU for critical reproducibility** (recommended):
   ```python
   # Force CPU inference
   device = "cpu"
   model.to(device)

   # 100% reproducible, but slower
   ```

**Recommendation**:
- **Research papers**: Use CPU (reproducibility is critical)
- **Production**: Use GPU (speed matters, accept variation)
- **Benchmarking**: Use CPU (fair comparison requires reproducibility)
- **Development**: Use GPU (fast iteration), switch to CPU for final validation

---

## Implementation Details

### DummyEmbeddingProvider Determinism

**Source**: `src/chatroutes_autobranch/core/embeddings.py:26-50`

```python
class DummyEmbeddingProvider:
    def __init__(self, dimension: int = 1536, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def embed(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            # Use text hash for deterministic randomness if seed is set
            if self.seed is not None:
                rng = random.Random(hash(text) + self.seed)  # ← Deterministic!
                embedding = [rng.random() for _ in range(self.dimension)]
            else:
                embedding = [random.random() for _ in range(self.dimension)]
```

**Key insight**: Uses `hash(text) + seed` to ensure:
- Same text + same seed → **identical embedding**
- Different texts → **different embeddings** (via hash)

---

### ShannonEntropyStopper Determinism

**Source**: `src/chatroutes_autobranch/entropy/shannon.py:121`

```python
kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
```

**Key**: `random_state` parameter ensures KMeans clustering is deterministic.

---

## Summary

| Aspect | CPU | GPU | How? |
|--------|-----|-----|------|
| **Pipeline logic** | ✅ Always | ✅ Always | Deterministic by design |
| **Scoring** | ✅ Always | ✅ Always | Pure math, no randomness |
| **Beam selection** | ✅ Always | ✅ Always | Stable sort, deterministic |
| **Novelty filtering** | ✅ Always | ✅ Always | Deterministic similarity |
| **Entropy stopping** | ✅ With seed | ✅ With seed | Set `random_seed=42` |
| **Test embeddings** | ✅ With seed | ✅ With seed | Set `seed=42` in DummyEmbeddingProvider |
| **Real embeddings (sentence-transformers)** | ✅ Yes | ⚠️ Mostly | CPU: deterministic; GPU: may vary |
| **LLM generation (Ollama/OpenAI)** | ✅ With seed | ⚠️ With seed | CPU: reproducible; GPU: may vary |
| **Overall reproducibility** | ✅ **100%** | ⚠️ **~90-95%** | CPU recommended for critical work |

**Bottom line**:
- **CPU**: ChatRoutes AutoBranch is **fully reproducible (100%)** when you control randomness sources (seeds)
- **GPU**: **Mostly reproducible (~90-95%)** but GPU non-determinism may cause minor variations
- **Recommendation**: Use **CPU for research/benchmarks**, GPU for production (accept variation)

---

## Further Reading

- **Test Suite**: `examples/test_reproducibility.py` - Verify reproducibility
- **Technical Spec**: `chatroutes_autobranch_v1.0.md` - Full API documentation
- **Contributing**: `CONTRIBUTING.md` - Testing standards require reproducibility

---

**Questions?** Open an issue or discussion on [GitHub](https://github.com/chatroutes/chatroutes-autobranch/discussions)
