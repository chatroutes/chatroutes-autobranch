# Llama 3 Integration Quick Start

## Option 1: Ollama (Recommended - Easiest)

### Installation

```bash
# 1. Install Ollama
# Windows: Download from https://ollama.ai
# Mac: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull Llama 3
ollama pull llama3

# 3. Install Python client
pip install ollama
```

### Usage

```python
from chatroutes_autobranch.branch_detection import LLMBranchParser
import ollama

def llama3_ollama(prompt: str) -> str:
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1}
    )
    return response['message']['content']

# Create parser
parser = LLMBranchParser(llm=llama3_ollama)

# Analyze text
conversation_text = """
User: I'm thinking about writing either a technical guide or a philosophical essay.
User: Actually, maybe I should do both in separate articles.
"""

branches = parser.parse(conversation_text)
print(f"Found {len(branches)} branches")
```

**Pros:**
- ✅ Free, unlimited
- ✅ Private (runs locally)
- ✅ Fast with GPU
- ✅ Easy to install

**Cons:**
- ❌ Requires ~4GB disk space
- ❌ Slower without GPU

---

## Option 2: Groq API (Fastest Cloud Option)

### Installation

```bash
# 1. Get API key from https://console.groq.com

# 2. Install client
pip install groq
```

### Usage

```python
from chatroutes_autobranch.branch_detection import LLMBranchParser
from groq import Groq

def llama3_groq(prompt: str) -> str:
    client = Groq(api_key="your-api-key-here")
    response = client.chat.completions.create(
        model="llama3-70b-8192",  # or llama3-8b-8192
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content

parser = LLMBranchParser(llm=llama3_groq)
branches = parser.parse(conversation_text)
```

**Pros:**
- ✅ Very fast (~1s)
- ✅ No local setup
- ✅ Access to 70B model
- ✅ Cheap ($0.10/1M tokens)

**Cons:**
- ❌ Requires API key
- ❌ Not private
- ❌ Costs money (minimal)

---

## Option 3: HTTP API (Ollama via Requests)

### Installation

```bash
# 1. Install Ollama (see Option 1)
ollama pull llama3

# 2. Install requests
pip install requests
```

### Usage

```python
from chatroutes_autobranch.branch_detection import LLMBranchParser
import requests

def llama3_api(prompt: str) -> str:
    response = requests.post(
        'http://localhost:11434/api/chat',
        json={
            'model': 'llama3',
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': False,
            'options': {'temperature': 0.1}
        }
    )
    return response.json()['message']['content']

parser = LLMBranchParser(llm=llama3_api)
branches = parser.parse(conversation_text)
```

---

## Complete Example: Three-Way Analysis

```python
from chatroutes_autobranch.branch_detection import (
    LLMBranchParser,
    ConversationFlowAnalyzer,
    ConversationTurn,
)
from chatroutes_autobranch.core.embeddings import DummyEmbeddingProvider
import ollama

# 1. Set up Llama 3
def llama3(prompt: str) -> str:
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1}
    )
    return response['message']['content']

# 2. Create your conversation
conversation = [
    ConversationTurn(
        id="1",
        speaker="user",
        content="I'm wondering if I should focus on philosophy or practical advice."
    ),
    ConversationTurn(
        id="2",
        speaker="user",
        content="""
BRANCH: Article Focus
OPTIONS:
1. Philosophy - Deep dive into meaning and value
2. Practical - Actionable advice for professionals
3. Both - Multi-article series
"""
    ),
]

# 3. Run three types of detection

# A) Explicit (deterministic, instant)
analyzer = ConversationFlowAnalyzer(
    embedding_provider=DummyEmbeddingProvider(dimension=384, seed=42),
    enable_explicit=True,
    enable_semantic=True,
)
results = analyzer.analyze(conversation)

print(f"Explicit branches: {len(results['explicit_branches'])}")
print(f"Semantic branches: {len(results['semantic_branches'])}")

# B) Llama 3 (AI-powered, ~2-5s)
parser = LLMBranchParser(llm=llama3)
conversation_text = "\n\n".join([f"[{t.speaker}]: {t.content}" for t in conversation])
llm_branches = parser.parse(conversation_text)

print(f"Llama 3 branches: {len(llm_branches)}")

# C) Combined results
total = (
    len(results['explicit_branches']) +
    len(results['semantic_branches']) +
    len(llm_branches)
)
print(f"\nTotal branches detected: {total}")
```

---

## When to Use Each Method

| Method | Speed | Cost | Privacy | Use Case |
|--------|-------|------|---------|----------|
| **Hybrid only** | Instant | Free | ✓ | Most cases, production |
| **+ Ollama** | 2-5s | Free | ✓ | Complex conversations, offline |
| **+ Groq** | ~1s | $0.10/1M | ✗ | Fast API, prototyping |

---

## Performance Comparison

```python
import time

# Benchmark
conversation_text = "Your conversation here..."

# 1. Hybrid (no LLM)
start = time.time()
results = analyzer.analyze(conversation)
print(f"Hybrid: {time.time() - start:.2f}s")
# Output: Hybrid: 0.05s

# 2. Llama 3 (Ollama)
start = time.time()
branches = parser.parse(conversation_text)
print(f"Llama 3: {time.time() - start:.2f}s")
# Output: Llama 3: 2.34s (with GPU)
```

---

## Troubleshooting

### Ollama Not Found

```bash
# Check if Ollama is installed
ollama --version

# Check if model is pulled
ollama list

# Pull Llama 3 if not present
ollama pull llama3
```

### Connection Error

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### Slow Performance

```bash
# Use smaller model
ollama pull llama3:8b

# Or use Groq for faster cloud inference
```

---

## Full Working Example

See `examples/llama3_branch_detection.py` for a complete working example with all methods!

```bash
python examples/llama3_branch_detection.py
```
