import json
import os
import re
import time
from pathlib import Path
from openai import OpenAI

# ==========================
# Configuration
# ==========================

MODEL = "gpt-4o-mini"

PROMPTS_PER_CALL = 10
CALLS_PER_CATEGORY = 13

OUTPUT_FILE = "synthetic_dataset.json"

CATEGORIES = [
    "factual_qa",
    "math_reasoning",
    "sentiment",
    "summarization",
    "ner",
    "debugging",
    "logical_reasoning",
    "code_generation",
]

# ==========================
# Fireworks Client
# ==========================

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

if api_key is None:
    raise RuntimeError("FIREWORKS_API_KEY not set")

if base_url is None:
    raise RuntimeError("FIREWORKS_BASE_URL not set")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# ==========================
# System Prompta
# ==========================

SYSTEM_PROMPT = """
You are creating a dataset for training an AI routing classifier.

Return ONLY valid JSON.

Return a JSON array.

Each element MUST have exactly this format:

{
  "prompt": "...",
  "category": "...",
  "difficulty": "easy"
}

or

{
  "prompt": "...",
  "category": "...",
  "difficulty": "hard"
}

Rules:

- Every prompt must belong ONLY to the requested category.
- category must exactly match the requested category.
- Generate diverse prompts.
- Mix short and long prompts.
- Mix beginner and advanced prompts.
- Roughly 50% easy and 50% hard.
- Never generate duplicate prompts.
- Never include answers.
- Never include markdown.
- Never explain anything.
- Return JSON only.
"""

dataset = []

# ==========================
# Generate
# ==========================

for category in CATEGORIES:

    print(f"\nGenerating {category}")

    for call in range(CALLS_PER_CATEGORY):

        user_prompt = f"""
Generate {PROMPTS_PER_CALL} unique prompts.

Category:

{category}

Return ONLY JSON.
"""

        try:

            response = client.chat.completions.create(
                model=MODEL,
                temperature=1.0,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            text = response.choices[0].message.content.strip()

            # Remove accidental markdown if present
            text = re.sub(r"^```json", "", text)
            text = re.sub(r"```$", "", text)
            text = text.strip()

            items = json.loads(text)

            dataset.extend(items)

            print(
                f"{category:20s} "
                f"Batch {call+1:02d}/{CALLS_PER_CATEGORY} "
                f"Total={len(dataset)}"
            )

        except Exception as e:
            print("Failed:", e)

        time.sleep(1)

# ==========================
# Remove duplicates
# ==========================

seen = set()
unique = []

for item in dataset:

    prompt = item["prompt"].strip()

    if prompt not in seen:
        seen.add(prompt)
        unique.append(item)

dataset = unique

# ==========================
# Save
# ==========================

with open(OUTPUT_FILE, "w") as f:
    json.dump(dataset, f, indent=2)

print("\nFinished!")
print("Saved", len(dataset), "examples")
print("File:", OUTPUT_FILE)