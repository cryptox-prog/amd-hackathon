#!/usr/bin/env python3

from router.infer_router import classify

examples = [
    "What is the capital of Australia?",
    "If a train travels 120 km in 2 hours, what is its average speed?",
    "I absolutely loved the movie. The acting was phenomenal!",
    "Summarize the following article in three sentences.",
    "Extract all named entities from: Elon Musk founded SpaceX in California.",
    "Why does this Python code throw an IndexError?",
    "Three switches control three bulbs in another room. How can you determine which switch controls which bulb with only one visit?",
    "Write a Python function to merge two sorted linked lists.",
]

print("=" * 80)
print("AMD Router Demo")
print("=" * 80)

for i, prompt in enumerate(examples, 1):
    result = classify(prompt)

    print(f"\nExample {i}")
    print("-" * 80)
    print(f"Prompt      : {prompt}")
    print(f"Category    : {result['category']}")
    print(f"Difficulty  : {result['difficulty']}")
    print(f"Confidence  : {result['confidence']:.4f}")

print("\nDone.")