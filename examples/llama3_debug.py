"""
Debug version: See exactly what Llama 3 returns.
"""

from chatroutes_autobranch.branch_detection import LLMBranchParser
import ollama
import json

def my_llm_debug(prompt: str) -> str:
    """Call Llama 3 and show the raw response."""
    print("\n" + "="*80)
    print("SENDING TO LLAMA 3:")
    print("="*80)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print()

    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1}
    )

    raw_response = response['message']['content']

    print("\n" + "="*80)
    print("RAW RESPONSE FROM LLAMA 3:")
    print("="*80)
    print(raw_response)
    print()

    return raw_response


# Create parser
parser = LLMBranchParser(llm=my_llm_debug)

# Test conversation with VERY CLEAR decision points
conversation = """
User: I'm writing an article. Should I use Flask or FastAPI for the examples?

User: Also, should I target beginners or advanced developers?

Options:
1. Flask - Simpler, better for beginners
2. FastAPI - Modern, better for advanced users
"""

print("="*80)
print("LLAMA 3 DEBUG TEST")
print("="*80)
print("\nTesting conversation:")
print(conversation)

print("\nAttempting to parse...")
branches, metadata = parser.parse_with_confidence(conversation)

print("\n" + "="*80)
print("PARSING RESULTS:")
print("="*80)
print(f"Success: {metadata['success']}")
print(f"Attempts: {metadata['attempts']}")
if metadata['error']:
    print(f"Error: {metadata['error']}")

print(f"\nBranches detected: {len(branches)}")

if branches:
    for i, branch in enumerate(branches, 1):
        print(f"\n{i}. {branch.type.upper()}")
        print(f"   Options:")
        for opt in branch.options:
            print(f"     - {opt.label}")
else:
    print("\nNo branches detected.")
    print("\nPossible reasons:")
    print("1. LLM didn't return valid JSON")
    print("2. LLM didn't identify decision points")
    print("3. Temperature too high (set to 0.1)")
    print("\nCheck the RAW RESPONSE above to diagnose.")

print("\n" + "="*80)
print("TIP: Compare the RAW RESPONSE to the expected JSON format.")
print("Expected format:")
print("""
{
  "branch_points": [
    {
      "id": "bp1",
      "type": "disjunction",
      "options": [
        {"id": "opt1", "label": "Flask", "span": "Flask"},
        {"id": "opt2", "label": "FastAPI", "span": "FastAPI"}
      ],
      "context": "Should I use Flask or FastAPI",
      "depends_on": []
    }
  ]
}
""")
print("="*80)