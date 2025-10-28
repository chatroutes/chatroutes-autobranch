"""
Simplest possible Llama 3 integration example.

Just 3 steps:
1. Install Ollama: https://ollama.ai
2. Run: ollama pull llama3
3. Run this script!
"""

from chatroutes_autobranch.branch_detection import LLMBranchParser

# Step 1: Define your Llama 3 function
def my_llm(prompt: str) -> str:
    """Call Llama 3 via Ollama."""
    try:
        import ollama
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except ImportError:
        return "ERROR: Install ollama with: pip install ollama"
    except Exception as e:
        return f"ERROR: {str(e)}\n\nMake sure Ollama is installed and run: ollama pull llama3"


# Step 2: Create parser
parser = LLMBranchParser(llm=my_llm)

# Step 3: Analyze conversation
conversation = """
User: I'm thinking about my article. Should I focus on philosophy or practical advice?

User: Actually, I think I'll do both. Let me plan a multi-article series:
- Article 1: Philosophy of human value
- Article 2: Practical advice for professionals
- Article 3: Company evolution

User: On second thought, maybe I should start with just the philosophical one.
"""

print("Analyzing conversation with Llama 3...\n")
branches = parser.parse(conversation)

print(f"Found {len(branches)} branches:\n")

for i, branch in enumerate(branches, 1):
    print(f"{i}. Type: {branch.type}")
    print(f"   Options:")
    for opt in branch.options:
        print(f"     - {opt.label}")
    print()

if not branches:
    print("No branches detected.")
    print("\nTip: The conversation might not have explicit decision points.")
    print("Try adding more structured choices, or use the hybrid analyzer!")
