# Google Colab Guide for ChatRoutes AutoBranch

**Last Updated**: 2025-01-22

---

## Table of Contents

1. [Overview](#overview)
2. [GPU vs CPU: Performance & Cost](#gpu-vs-cpu-performance--cost)
3. [How Model Downloads Work](#how-model-downloads-work)
4. [Quick Start](#quick-start)
5. [Cost Analysis](#cost-analysis)
6. [Optimization Tips](#optimization-tips)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Colab notebook (`notebooks/creative_writing_colab.ipynb`) allows you to run ChatRoutes AutoBranch in Google Colab with:

- ✅ **100% FREE option** (CPU runtime)
- ⚡ **Optional GPU acceleration** (free tier or paid)
- 🔄 **Automatic model downloads**
- 📊 **Performance benchmarks**

---

## GPU vs CPU: Performance & Cost

### Performance Comparison

| Task | CPU (Free) | GPU Free (T4) | GPU Pro (V100) | Speedup |
|------|-----------|---------------|----------------|---------|
| **LLM Generation** (llama3.1:8b) | 40-50s | 1-3s | 0.5-1s | 20-40x |
| **Embeddings** (1000 texts) | 10-15s | 2-3s | 1-2s | 5-10x |
| **Total (4 scenarios)** | 30-40 min | 5-10 min | 3-5 min | 6-10x |

### Cost Comparison

| Runtime Type | Cost | GPU Hours | Best For |
|--------------|------|-----------|----------|
| **CPU** | $0 | N/A | Learning, testing, unlimited use |
| **GPU (Free Tier)** | $0 | Limited (~15-20 hrs/week) | Quick demos |
| **Colab Pro** | $10/month | ~100 hrs/month | Regular development |
| **Colab Pro+** | $50/month | ~500 hrs/month | Heavy usage |

### Recommendation

**Start with CPU (free)** for:
- ✅ Learning the library
- ✅ Testing small examples
- ✅ Development (no time pressure)

**Upgrade to GPU** when:
- ⚡ You need fast iteration
- 🚀 Running multiple experiments
- 💼 Production demos

---

## How Model Downloads Work

### 1. Ollama Models (LLMs)

**Location**: `/usr/share/ollama/.ollama/models/`

**Behavior**:
```python
# First time in session
!ollama pull llama3.1:8b
# ⏱️ Downloads 4.9 GB (~2-5 minutes)
# ✅ Model available for entire session

# After runtime disconnect
# ❌ Model deleted
# 🔄 Must re-download in new session
```

**Download Times** (on Colab):
| Model | Size | Download Time | RAM Required |
|-------|------|---------------|--------------|
| llama3.1:8b | 4.9 GB | 2-3 min | 8 GB |
| qwen3:14b | 9.3 GB | 4-6 min | 16 GB |
| gpt-oss:20b | 13 GB | 6-8 min | 24 GB |

**Storage Location**:
- Colab has ~100-200 GB disk space
- Models stored in `/usr/share/ollama/`
- Deleted when runtime disconnects (~12 hours idle, ~24 hours active)

### 2. Sentence-Transformers (Embeddings)

**Location**: `/root/.cache/huggingface/hub/`

**Behavior**:
```python
from sentence_transformers import SentenceTransformer

# First time in session
model = SentenceTransformer('all-mpnet-base-v2')
# ⏱️ Downloads 420 MB (~1-2 minutes)
# ✅ Cached for entire session

# Subsequent uses
model = SentenceTransformer('all-mpnet-base-v2')
# ⚡ Loads from cache (instant!)

# After runtime disconnect
# ❌ Cache deleted
# 🔄 Must re-download in new session
```

**Download Times**:
| Model | Size | Download Time | Dimension |
|-------|------|---------------|-----------|
| all-mpnet-base-v2 | 420 MB | 1-2 min | 768D |
| jina-embeddings-v2-base-en | 560 MB | 2-3 min | 768D |
| bge-large-en-v1.5 | 1.2 GB | 3-4 min | 1024D |

### 3. chatroutes-autobranch Library

**Installation**:
```python
!pip install chatroutes-autobranch
# ⏱️ ~30 seconds
# ✅ Persists for session
```

### Total First-Run Setup Time

| Components | CPU | GPU |
|------------|-----|-----|
| Install dependencies | 2 min | 2 min |
| Download Ollama model (llama3.1:8b) | 3 min | 3 min |
| Download embedding models (auto) | 1-2 min | 1-2 min |
| **Total setup** | **6-7 min** | **6-7 min** |
| **Then run demo** | 30-40 min | 5-10 min |

---

## Quick Start

### Option 1: Open in Colab (Easiest)

1. **Open the notebook**:
   - Click the "Open in Colab" badge in the notebook
   - Or visit: `https://colab.research.google.com/github/yourusername/chatroutes-autobranch/blob/main/notebooks/creative_writing_colab.ipynb`

2. **Choose runtime**:
   ```
   Runtime → Change runtime type
   - CPU (free, unlimited) ← Recommended for learning
   - GPU (free tier, limited) ← Faster
   ```

3. **Run all cells**:
   ```
   Runtime → Run all
   ```

4. **Wait for completion**:
   - Setup: 6-7 minutes
   - Demo: 5-10 min (GPU) or 30-40 min (CPU)

### Option 2: Manual Setup

```bash
# 1. Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama server
import subprocess
ollama_process = subprocess.Popen(['ollama', 'serve'])

# 3. Download model
!ollama pull llama3.1:8b

# 4. Install Python packages
!pip install chatroutes-autobranch sentence-transformers

# 5. Run demo
!python examples/creative_writting_usage.py
```

---

## Cost Analysis

### Scenario: Running 4 Creative Writing Demos Daily

| Runtime | Time/Run | Cost/Month | Breakeven |
|---------|----------|------------|-----------|
| **CPU (Free)** | 35 min | $0 | Always free |
| **GPU Free Tier** | 7 min | $0* | Use free hours first |
| **Colab Pro** | 7 min | $10 | If >100 hrs GPU needed |

*Free tier: ~15-20 GPU hours/week (~2-3 hours/day)

### Cost Calculation Examples

**Example 1: Student (learning)**
- Usage: 1-2 hours/day
- Recommendation: **CPU (free)**
- Cost: **$0/month**

**Example 2: Developer (testing)**
- Usage: 2-3 runs/day (~20 min GPU each)
- Recommendation: **GPU Free Tier**
- Cost: **$0/month** (stays within free tier)

**Example 3: Researcher (heavy use)**
- Usage: 10+ runs/day
- Recommendation: **Colab Pro**
- Cost: **$10/month** (worth the time savings)

---

## Optimization Tips

### 1. Reduce Setup Time

**Use smaller models**:
```python
MODEL = "llama3.1:8b"  # 4.9 GB, fastest
```

**Use lightweight embeddings**:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')  # 80 MB, 10x faster download
```

### 2. Persist Models Across Sessions (Advanced)

**Mount Google Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')

# Symlink Ollama models to Drive
!mkdir -p /content/drive/MyDrive/ollama_models
!ln -s /content/drive/MyDrive/ollama_models /usr/share/ollama/.ollama/models

# Now models persist across sessions!
```

**Cache embeddings to Drive**:
```python
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/huggingface_cache'
```

### 3. Optimize for GPU

**Use larger batch sizes**:
```python
# Embeddings
embeddings = model.encode(texts, batch_size=64)  # GPU: 64, CPU: 8

# LLM (Ollama auto-optimizes)
```

**Monitor GPU usage**:
```python
!nvidia-smi
```

### 4. Reduce Runtime

**Run fewer scenarios**:
```python
# Test with 1 scenario first
scenario_1_ai_memory(model_name=model_name)
# Skip others for quick testing
```

**Generate fewer candidates**:
```python
parent, candidates = generate_creative_candidates(
    prompt=prompt,
    model_name=model_name,
    n=5,  # Default: 10-15, use 5 for faster testing
    temperature=1.2
)
```

---

## Troubleshooting

### Problem 1: "Runtime disconnected"

**Causes**:
- Idle for >90 minutes
- Free GPU quota exceeded
- Tab closed

**Solutions**:
1. **Prevent idle disconnect**:
   ```javascript
   // Run in browser console
   function KeepAlive() {
     console.log("Keeping alive...");
     document.querySelector("colab-connect-button").click();
   }
   setInterval(KeepAlive, 60000);
   ```

2. **Use Colab Pro**: Longer timeouts, more stable

3. **Save checkpoints**: Export results periodically

### Problem 2: "Out of Memory"

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. **Use smaller model**:
   ```python
   MODEL = "llama3.1:8b"  # Instead of qwen3:14b
   ```

2. **Restart runtime**:
   ```
   Runtime → Restart runtime
   ```

3. **Upgrade to Colab Pro** (more RAM/VRAM)

### Problem 3: "Ollama server not responding"

**Symptoms**:
```
Connection refused: localhost:11434
```

**Solutions**:
1. **Check if running**:
   ```python
   import requests
   response = requests.get('http://localhost:11434/api/tags')
   print(response.status_code)  # Should be 200
   ```

2. **Restart server**:
   ```python
   ollama_process.kill()
   ollama_process = subprocess.Popen(['ollama', 'serve'])
   ```

3. **Check logs**:
   ```python
   print(ollama_process.stderr.read())
   ```

### Problem 4: "Model download failed"

**Symptoms**:
```
Error: failed to pull model
```

**Solutions**:
1. **Check disk space**:
   ```bash
   !df -h /
   ```

2. **Try smaller model**:
   ```python
   MODEL = "llama3.1:8b"  # Only 4.9 GB
   ```

3. **Restart and retry**:
   ```
   Runtime → Restart runtime
   ```

### Problem 5: "Slow generation on GPU"

**Symptoms**:
- GPU available but still slow (~40s per response)

**Causes**:
- Ollama not using GPU
- Model not loaded to GPU

**Solutions**:
1. **Verify GPU detection**:
   ```bash
   !nvidia-smi
   ```

2. **Check Ollama GPU usage**:
   ```bash
   !nvidia-smi dmon -s u
   # Run while generating
   ```

3. **Restart Ollama with GPU**:
   ```python
   ollama_process.kill()
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'
   ollama_process = subprocess.Popen(['ollama', 'serve'])
   ```

---

## Advanced: GPU Performance Tuning

### 1. Check GPU Utilization

```python
import subprocess
import threading

def monitor_gpu():
    """Monitor GPU usage in background."""
    while True:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader'],
            capture_output=True, text=True
        )
        print(f"GPU Usage: {result.stdout.strip()}")
        time.sleep(5)

# Start monitoring
thread = threading.Thread(target=monitor_gpu, daemon=True)
thread.start()
```

### 2. Optimize Ollama for GPU

```python
# Set environment variables before starting Ollama
import os
os.environ['OLLAMA_NUM_GPU'] = '1'  # Use 1 GPU
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Keep 1 model in VRAM

ollama_process = subprocess.Popen(['ollama', 'serve'])
```

### 3. Batch Processing

```python
# Generate multiple candidates in parallel (if Ollama supports)
# Currently Ollama generates sequentially, but embeddings can batch:

from concurrent.futures import ThreadPoolExecutor

def generate_candidate(i):
    # Generate single candidate
    pass

# Generate 10 candidates with 3 parallel workers
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(generate_candidate, range(10)))
```

---

## FAQ

### Q: Will my models persist across sessions?
**A**: No, Colab's runtime is ephemeral. Models must be re-downloaded each session (~6-7 minutes setup).

### Q: Can I use my own GPU locally instead?
**A**: Yes! The same code works locally. Install Ollama locally for persistent models.

### Q: How much does GPU acceleration cost?
**A**:
- Free tier: $0 (limited hours)
- Colab Pro: $10/month
- Colab Pro+: $50/month

### Q: Is CPU mode really free and unlimited?
**A**: Yes! CPU runtime has no usage limits, just slower performance.

### Q: Can I use Colab for production?
**A**: Not recommended. Colab is for research/development. For production:
- Use cloud GPU APIs (RunPod, Vast.ai)
- Or deploy Ollama on your own server

### Q: Which model should I use?
**A**:
- **Testing**: llama3.1:8b (fast, good quality)
- **Production**: qwen3:14b (best quality)
- **Research**: gpt-oss:20b (highest quality)

### Q: How do I save results?
**A**:
```python
# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save results
import json
with open('/content/drive/MyDrive/results.json', 'w') as f:
    json.dump(results, f)
```

---

## Next Steps

1. **Try the notebook**: Open `notebooks/creative_writing_colab.ipynb`
2. **Read the main docs**: See `README.md` for library details
3. **Explore examples**: Check `examples/` for more use cases
4. **Join community**: Report issues on GitHub

---

**Made with ❤️ using ChatRoutes AutoBranch**
