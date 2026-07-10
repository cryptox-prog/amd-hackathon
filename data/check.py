#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path

DATASET = Path("data/synthetic_dataset.json")

CLASSES = [
    "factual_qa_easy",
    "factual_qa_hard",
    "math_reasoning_easy",
    "math_reasoning_hard",
    "sentiment_easy",
    "sentiment_hard",
    "summarization_easy",
    "summarization_hard",
    "ner_easy",
    "ner_hard",
    "debugging_easy",
    "debugging_hard",
    "logical_reasoning_easy",
    "logical_reasoning_hard",
    "code_generation_easy",
    "code_generation_hard",
]

with open(DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)

counter = Counter()

for item in dataset:
    category = item.get("category")
    difficulty = item.get("difficulty")

    label = f"{category}_{difficulty}"

    if label in CLASSES:
        counter[label] += 1
    else:
        print(f"Invalid label: {label}")

print("=" * 45)
print(f"Total examples: {len(dataset)}")
print("=" * 45)

for label in CLASSES:
    print(f"{label:30s} {counter[label]:4d}")

print("=" * 45)

# Optional summary
easy = sum(counter[c] for c in CLASSES if c.endswith("_easy"))
hard = sum(counter[c] for c in CLASSES if c.endswith("_hard"))

print(f"Easy examples : {easy}")
print(f"Hard examples : {hard}")