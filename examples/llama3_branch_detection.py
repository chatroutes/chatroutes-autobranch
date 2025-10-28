"""
Using Llama 3 for advanced branch detection in conversations.

This example shows how to:
1. Use Llama 3 via Ollama (local) or API (remote)
2. Integrate with LLMBranchParser
3. Combine with hybrid analyzer for best results
"""

import json
from typing import Callable

from chatroutes_autobranch.branch_detection import (
    LLMBranchParser,
    ConversationFlowAnalyzer,
    ConversationTurn,
)
from chatroutes_autobranch.core.embeddings import DummyEmbeddingProvider


# ============================================================================
# METHOD 1: Ollama (Local - Recommended)
# ============================================================================

def llama3_ollama(prompt: str) -> str:
    """
    Call Llama 3 via Ollama (local).

    Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull Llama 3: ollama pull llama3
    3. Install Python client: pip install ollama

    Args:
        prompt: The prompt to send to Llama 3.

    Returns:
        Response string from Llama 3.
    """
    try:
        import ollama

        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.1,  # Low temperature for structured output
                'num_predict': 2000,  # Max tokens
            }
        )

        return response['message']['content']

    except ImportError:
        return """
ERROR: ollama package not installed.
Install with: pip install ollama
Then run: ollama pull llama3
"""
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================================================================
# METHOD 2: Ollama API (HTTP)
# ============================================================================

def llama3_ollama_api(prompt: str) -> str:
    """
    Call Llama 3 via Ollama HTTP API.

    Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull Llama 3: ollama pull llama3
    3. Start server: ollama serve (usually runs automatically)

    Args:
        prompt: The prompt to send to Llama 3.

    Returns:
        Response string from Llama 3.
    """
    try:
        import requests

        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model': 'llama3',
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False,
                'options': {
                    'temperature': 0.1,
                }
            }
        )

        response.raise_for_status()
        data = response.json()
        return data['message']['content']

    except ImportError:
        return "ERROR: requests package not installed. Install with: pip install requests"
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================================================================
# METHOD 3: Groq API (Fast Cloud Inference)
# ============================================================================

def llama3_groq(prompt: str, api_key: str) -> str:
    """
    Call Llama 3 via Groq API (very fast cloud inference).

    Prerequisites:
    1. Get API key: https://console.groq.com
    2. Install client: pip install groq

    Args:
        prompt: The prompt to send to Llama 3.
        api_key: Your Groq API key.

    Returns:
        Response string from Llama 3.
    """
    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama3-70b-8192",  # or "llama3-8b-8192" for faster/cheaper
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    except ImportError:
        return "ERROR: groq package not installed. Install with: pip install groq"
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================================================================
# METHOD 4: Hugging Face Transformers (Local)
# ============================================================================

def llama3_huggingface(prompt: str) -> str:
    """
    Call Llama 3 via Hugging Face Transformers (local).

    Prerequisites:
    1. pip install transformers torch
    2. Requires GPU for reasonable speed
    3. Model will be downloaded on first run (~16GB)

    Args:
        prompt: The prompt to send to Llama 3.

    Returns:
        Response string from Llama 3.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # Load model (cached after first run)
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate
        outputs = model.generate(
            input_ids,
            max_new_tokens=2000,
            temperature=0.1,
            do_sample=True,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except ImportError:
        return "ERROR: transformers or torch not installed. Install with: pip install transformers torch"
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================================================================
# Example Usage
# ============================================================================

def main():
    print("=" * 80)
    print("LLAMA 3 BRANCH DETECTION EXAMPLE")
    print("=" * 80)

    # Your conversation
    conversation = [
        ConversationTurn(
            id="1",
            speaker="user",
            content="I'm wondering whether to focus on philosophy or practical advice for my article."
        ),
        ConversationTurn(
            id="2",
            speaker="user",
            content="Actually, I want to do both. Maybe I should write multiple articles."
        ),
        ConversationTurn(
            id="3",
            speaker="user",
            content="Let me think about the structure: one on creativity definitions, another on professions, and a third on company evolution."
        ),
    ]

    print("\nConversation to analyze:")
    for turn in conversation:
        print(f"  [{turn.speaker}] {turn.content[:80]}...")

    # ========================================================================
    # STEP 1: Choose your Llama 3 provider
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: Choose Llama 3 Provider")
    print("=" * 80)
    print("""
Available options:
1. Ollama (local, recommended) - Fast, free, private
2. Groq API (cloud) - Very fast, requires API key
3. Hugging Face (local) - Requires GPU, fully offline

For this example, we'll use Ollama (option 1).
If Ollama is not installed, it will show an error with instructions.
""")

    # Choose your LLM function
    llm_function = llama3_ollama  # Change to llama3_groq or llama3_huggingface as needed

    # ========================================================================
    # STEP 2: Create LLM Branch Parser
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: Create LLM Branch Parser")
    print("=" * 80)

    parser = LLMBranchParser(
        llm=llm_function,
        temperature=0.1,
        max_retries=2,
    )

    print("\nParser created with Llama 3")
    print("Temperature: 0.1 (low = more deterministic)")
    print("Max retries: 2")

    # ========================================================================
    # STEP 3: Parse conversation with LLM
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: Parse Conversation with Llama 3")
    print("=" * 80)
    print("\nCalling Llama 3... (this may take a few seconds)")

    # Convert conversation to text
    conversation_text = "\n\n".join([
        f"[{turn.speaker}]: {turn.content}"
        for turn in conversation
    ])

    llm_branches, metadata = parser.parse_with_confidence(conversation_text)

    print(f"\nResult: {'SUCCESS' if metadata['success'] else 'FAILED'}")
    print(f"Attempts: {metadata['attempts']}")
    if metadata['error']:
        print(f"Error: {metadata['error']}")

    print(f"\nLlama 3 detected {len(llm_branches)} branches:")
    for i, branch in enumerate(llm_branches, 1):
        print(f"\n{i}. {branch.type.upper()} (ID: {branch.id})")
        print(f"   Options ({len(branch.options)}):")
        for opt in branch.options:
            print(f"     - {opt.label}")

    # ========================================================================
    # STEP 4: Combine with Hybrid Analyzer
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 4: Combine with Hybrid Analyzer")
    print("=" * 80)
    print("\nNow combining Llama 3 results with hybrid detection...")

    # Create hybrid analyzer
    analyzer = ConversationFlowAnalyzer(
        embedding_provider=DummyEmbeddingProvider(dimension=384, seed=42),
        enable_explicit=True,
        enable_semantic=True,
    )

    # Analyze with hybrid approach
    hybrid_results = analyzer.analyze(conversation)

    print("\nHybrid Detection Results:")
    print(f"  Explicit branches: {len(hybrid_results['explicit_branches'])}")
    print(f"  Semantic branches: {len(hybrid_results['semantic_branches'])}")
    print(f"  Llama 3 branches: {len(llm_branches)}")

    # Combine all results
    total_branches = (
        len(hybrid_results['explicit_branches']) +
        len(hybrid_results['semantic_branches']) +
        len(llm_branches)
    )

    print(f"\n  TOTAL BRANCHES: {total_branches}")

    # ========================================================================
    # STEP 5: Analysis Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 5: Complete Analysis Summary")
    print("=" * 80)

    print("\n1. EXPLICIT BRANCHES (User-marked):")
    if hybrid_results['explicit_branches']:
        for branch in hybrid_results['explicit_branches']:
            print(f"   - {branch.id}: {branch.option_count} options")
    else:
        print("   None detected")

    print("\n2. SEMANTIC BRANCHES (Pattern-based):")
    if hybrid_results['semantic_branches']:
        for branch in hybrid_results['semantic_branches']:
            print(f"   - {branch.branch_type}: {branch.description[:60]}...")
    else:
        print("   None detected")

    print("\n3. LLM BRANCHES (Llama 3):")
    if llm_branches:
        for branch in llm_branches:
            print(f"   - {branch.type}: {branch.option_count} options")
    else:
        print("   None detected")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
Best Practice:
1. Use hybrid analyzer for fast, reliable detection
2. Add Llama 3 for complex, ambiguous cases
3. Mark critical branches explicitly for 100% confidence

Performance:
- Hybrid: <1s (no LLM)
- Llama 3 (Ollama): ~2-5s (local GPU)
- Llama 3 (Groq): ~1s (cloud, fast)

Cost:
- Hybrid: Free, unlimited
- Llama 3 (Ollama): Free, unlimited
- Llama 3 (Groq): $0.10/1M tokens (cheap)
""")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
