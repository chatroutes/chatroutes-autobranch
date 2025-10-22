# GPU vs CPU Cheat Sheet - Quick Reference

## ⚡ Quick Answer

**Q: Does inference require GPU?**
**A: No, but GPU is 20-40x faster.**

---

## 📊 Speed Comparison

| Model | CPU | GPU Free (T4) | GPU Pro (V100) |
|-------|-----|---------------|----------------|
| llama3.1:8b | 40s | 1-3s | 0.5-1s |
| qwen3:14b | 80s | 3-5s | 1-2s |
| gpt-oss:20b | 120s | 5-8s | 2-3s |
| **Embeddings** (1000 texts) | 10s | 2s | 1s |

---

## 💰 Cost Comparison

| Setup | Cost | Best For |
|-------|------|----------|
| **CPU (Local)** | $0 | Development, unlimited use |
| **GPU (Free Colab)** | $0 | Quick demos (~15 hrs/week) |
| **GPU (Colab Pro)** | $10/mo | Regular use (~100 hrs/mo) |
| **GPU (Colab Pro+)** | $50/mo | Heavy use (~500 hrs/mo) |

---

## 🎯 Decision Matrix

### Use CPU If:
- ✅ Learning/testing
- ✅ No time pressure
- ✅ Unlimited usage needed
- ✅ $0 budget

### Use GPU If:
- ⚡ Need fast iteration
- ⚡ Running many experiments
- ⚡ Production demos
- ⚡ Time > Money

---

## 🔄 Local vs Colab

| Feature | Local (CPU) | Colab (CPU) | Colab (GPU) |
|---------|-------------|-------------|-------------|
| **Speed** | 40s/response | 40s/response | 1-3s/response |
| **Cost** | $0 | $0 | $0-50/mo |
| **Setup** | One-time | Every session | Every session |
| **Storage** | Persistent | Ephemeral | Ephemeral |
| **Limit** | None | None | GPU hours |

---

## 📥 Model Downloads

### Local
```bash
ollama pull llama3.1:8b  # Once
# ✅ Persists forever
```

### Colab
```python
!ollama pull llama3.1:8b  # Every session
# ❌ Lost after ~12-24 hours
# 🔄 Re-download: 2-5 minutes
```

**Solution**: Cache to Google Drive (see COLAB_GUIDE.md)

---

## ⏱️ Total Runtime (4 Scenarios)

| Setup | Generation | Pipeline | Total |
|-------|-----------|----------|-------|
| **CPU** | 35 min | 1 min | ~36 min |
| **GPU** | 2 min | 0.5 min | ~2.5 min |

**Speedup**: ~14x faster with GPU

---

## 🚀 Quick Start Commands

### Local (CPU)
```bash
# One-time setup
ollama pull llama3.1:8b
pip install chatroutes-autobranch sentence-transformers

# Run anytime
python examples/creative_writting_usage.py
```

### Colab (CPU)
```python
# Every session
!curl -fsSL https://ollama.com/install.sh | sh
!ollama serve &
!ollama pull llama3.1:8b
!pip install chatroutes-autobranch sentence-transformers
!python examples/creative_writting_usage.py
```

### Colab (GPU)
```python
# Same as CPU, but:
Runtime → Change runtime type → GPU (T4)
# Then run same commands
# ⚡ 20-40x faster!
```

---

## 💡 Optimization Tips

### Faster Downloads
```python
MODEL = "llama3.1:8b"  # 4.9 GB (fastest)
# vs
MODEL = "qwen3:14b"    # 9.3 GB
MODEL = "gpt-oss:20b"  # 13 GB
```

### Faster Testing
```python
n=5  # Generate 5 candidates instead of 10-15
# Cuts time in half
```

### Cache Models (Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
os.environ['HF_HOME'] = '/content/drive/MyDrive/cache'
# ✅ Persists across sessions
```

---

## 🐛 Common Issues

### "Too slow on CPU"
→ Expected! 40s/response is normal
→ Use GPU for speed
→ Or use smaller model

### "GPU quota exceeded"
→ Switch to CPU (free, unlimited)
→ Or upgrade to Colab Pro

### "Model not found"
→ Re-download: `ollama pull MODEL_NAME`
→ Models don't persist in Colab

### "Out of memory"
→ Use smaller model: `llama3.1:8b`
→ Restart runtime
→ Upgrade to Colab Pro

---

## 📊 Real-World Performance (Your Tests)

**Your Machine (CPU)**:
- llama3.1:8b: ~43s per response ✅
- Total time: 35-40 minutes ✅
- 100% FREE ✅

**Expected on GPU**:
- llama3.1:8b: ~1-3s per response ⚡
- Total time: 5-10 minutes ⚡
- Cost: $0-10/month 💰

---

## 🎓 Recommendation

**For Learning** → Start with CPU (local or Colab)
- ✅ Free
- ✅ Unlimited
- ✅ No setup hassle

**For Production** → Use GPU (Colab Pro or Cloud)
- ⚡ Fast
- ⚡ Efficient
- ⚡ Worth the cost

---

## 📚 Files Created

1. **`notebooks/creative_writing_colab.ipynb`** - Full Colab notebook
2. **`COLAB_GUIDE.md`** - 60+ page guide
3. **`COLAB_NOTEBOOK_README.md`** - Quick start
4. **`GPU_VS_CPU_CHEATSHEET.md`** - This file

---

## ✅ Bottom Line

| Question | Answer |
|----------|--------|
| **Need GPU?** | No, but 20-40x faster |
| **CPU cost?** | $0 forever |
| **GPU cost?** | $0-50/month |
| **Best for learning?** | CPU |
| **Best for production?** | GPU |
| **Models persist in Colab?** | No (re-download each session) |
| **Colab setup time?** | 6-7 minutes |
| **Total runtime (CPU)?** | 35-40 minutes |
| **Total runtime (GPU)?** | 5-10 minutes |

**Verdict**: Start with CPU (free), upgrade to GPU when speed matters!

---

Made with ❤️ using ChatRoutes AutoBranch
