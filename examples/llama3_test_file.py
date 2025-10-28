"""
Simple test that writes output to file to avoid terminal truncation.
"""

import sys
from chatroutes_autobranch.branch_detection import LLMBranchParser
import ollama

# Redirect output to file
output_file = open("llama3_debug_output.txt", "w", encoding="utf-8")
sys.stdout = output_file
sys.stderr = output_file

try:
    print("="*80)
    print("Testing Llama 3 Integration")
    print("="*80)

    def my_llm(prompt: str) -> str:
        print("\n[1] Calling Llama 3...")
        print(f"[2] Prompt length: {len(prompt)} chars")

        try:
            response = ollama.chat(
                model='llama3',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 1000}
            )

            raw = response['message']['content']
            print(f"[3] Response length: {len(raw)} chars")
            print(f"[4] First 200 chars: {raw[:200]}")
            print(f"[5] Last 200 chars: {raw[-200:]}")
            print("\n[6] FULL RESPONSE:")
            print(raw)

            return raw

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return f"ERROR: {str(e)}"

    # Simple test
    conversation = "User: Should I use Flask or FastAPI?"

    print("\n[7] Creating parser...")
    parser = LLMBranchParser(llm=my_llm)

    print("\n[8] Parsing conversation...")
    branches, meta = parser.parse_with_confidence(conversation)

    print("\n[9] RESULTS:")
    print(f"    Success: {meta['success']}")
    print(f"    Error: {meta.get('error', 'None')}")
    print(f"    Branches: {len(branches)}")

    for i, b in enumerate(branches):
        print(f"\n    Branch {i+1}:")
        print(f"      Type: {b.type}")
        print(f"      Options: {len(b.options)}")
        for opt in b.options:
            print(f"        - {opt.label}")

finally:
    output_file.close()
    # Restore stdout
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    print("\n" + "="*80)
    print("Output saved to: llama3_debug_output.txt")
    print("="*80)
    print("\nTo view results:")
    print("  type llama3_debug_output.txt")
    print("\nOr open the file in your editor")
