# ChatRoutes AutoBranch - Colab Notebooks

Interactive Google Colab notebooks to learn and experiment with ChatRoutes AutoBranch.

---

## 📓 Available Notebooks

### 1. Getting Started Demo (Recommended for Beginners)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chatroutes/chatroutes-autobranch/blob/master/notebooks/getting_started_demo.ipynb)

**Perfect for**: First-time users, video demos, quick introduction

**What you'll learn**:
- Installation and setup (2 minutes)
- Basic beam search examples
- Multi-strategy scoring
- Novelty filtering (duplicate removal)
- Complete pipeline with budget control

**Time**: ~5 minutes
**Requirements**: None (runs entirely in Colab)
**Dependencies**: Only chatroutes-autobranch (uses DummyEmbeddingProvider)

**Use this notebook when**:
- You're new to ChatRoutes AutoBranch
- You want a quick overview
- You're creating tutorial videos
- You need simple, clear examples

---

### 2. Creative Writing Scenario (Advanced)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chatroutes/chatroutes-autobranch/blob/master/notebooks/creative_writing_colab.ipynb)

**Perfect for**: Real-world scenarios, advanced users, production examples

**What you'll learn**:
- Full LLM integration with Ollama
- Multi-turn branching (tree exploration)
- Real embedding models (sentence-transformers)
- GPU vs CPU performance comparison
- Complete creative writing assistant

**Time**: ~30-40 minutes (CPU), ~5-10 minutes (GPU)
**Requirements**: Ollama installation, larger models
**Dependencies**: chatroutes-autobranch, sentence-transformers, ollama

**Use this notebook when**:
- You want to see a real-world application
- You need end-to-end LLM integration
- You're building a production system
- You want to test on GPU/CPU

---

## 🎯 Which Notebook Should You Use?

| Goal | Notebook | Why |
|------|----------|-----|
| **Learn the basics** | Getting Started Demo | Simple, fast, no complex setup |
| **Video tutorial** | Getting Started Demo | Clear examples, easy to explain |
| **Quick test** | Getting Started Demo | Runs in 5 minutes |
| **Real LLM integration** | Creative Writing | Shows Ollama integration |
| **Production example** | Creative Writing | Complete use case |
| **GPU performance** | Creative Writing | Tests on real hardware |

---

## 🚀 Quick Start

### Option 1: Getting Started Demo (Recommended)

1. Click the Colab badge above
2. Run all cells (Runtime → Run all)
3. Follow along with the examples
4. Total time: ~5 minutes

### Option 2: Creative Writing Scenario

1. Click the Colab badge above
2. Choose runtime: CPU (free, slow) or GPU (faster, may require Colab Pro)
3. Run all cells (Runtime → Run all)
4. Wait for Ollama installation and model download
5. Explore the 4 creative writing scenarios
6. Total time: ~30-40 minutes (CPU), ~5-10 minutes (GPU)

---

## 📚 Learning Path

### Beginner (Start Here!)

1. **Getting Started Demo** - Learn the fundamentals
   - Beam search
   - Scoring strategies
   - Novelty filtering
   - Budget management

### Intermediate

2. **Creative Writing Scenario** - See it in action
   - LLM integration (Ollama)
   - Multi-turn branching
   - Real embeddings
   - Performance tuning

### Advanced

3. **Build Your Own** - Apply to your use case
   - Check [examples/](../examples/) folder
   - Read [QUICKSTART.md](../QUICKSTART.md)
   - Review [API docs](../README.md)

---

## 💡 Tips

### For Video Demos
- Use **Getting Started Demo**
- Run all cells before recording
- Focus on one example per video segment
- Show the visual outputs

### For Learning
- Start with **Getting Started Demo**
- Modify the examples to experiment
- Try different parameters (k, weights, thresholds)
- Then move to **Creative Writing** for real integration

### For Production
- Study **Creative Writing Scenario**
- Adapt the patterns to your LLM
- Use the budget management features
- Test on both CPU and GPU

---

## 🐛 Troubleshooting

### Getting Started Demo

**Issue**: Import errors
**Fix**: Make sure you ran the installation cell first

**Issue**: No output shown
**Fix**: Run cells in order from top to bottom

### Creative Writing Scenario

**Issue**: Ollama download slow
**Fix**: Use smaller model (llama3.1:8b instead of qwen3:14b)

**Issue**: Out of memory
**Fix**: Restart runtime, use smaller model, or upgrade to Colab Pro

**Issue**: GPU not available
**Fix**: Runtime → Change runtime type → GPU

---

## 📧 Get Help

- **Issues**: https://github.com/chatroutes/chatroutes-autobranch/issues
- **Discussions**: https://github.com/chatroutes/chatroutes-autobranch/discussions
- **Email**: support@chatroutes.com

---

## 📦 Installation (Local)

Want to run these examples locally instead of Colab?

```bash
# Install the package
pip install chatroutes-autobranch

# For Getting Started Demo
pip install chatroutes-autobranch

# For Creative Writing Scenario
pip install chatroutes-autobranch[hf]  # Includes sentence-transformers
```

Then download the notebooks and run in Jupyter:

```bash
git clone https://github.com/chatroutes/chatroutes-autobranch.git
cd chatroutes-autobranch/notebooks
jupyter notebook
```

---

**Happy learning!** 🎓
