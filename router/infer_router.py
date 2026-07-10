#!/usr/bin/env python3

from pathlib import Path

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

CHECKPOINT = Path("checkpoints/router")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(CHECKPOINT)

model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT)
model.to(device)
model.eval()


def classify(prompt: str):
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)

    predicted_label = model.config.id2label[prediction.item()]

    # Split "math_reasoning_hard" -> ("math_reasoning", "hard")
    category, difficulty = predicted_label.rsplit("_", 1)

    return {
        "category": category,
        "difficulty": difficulty,
        "confidence": round(confidence.item(), 4),
    }


if __name__ == "__main__":

    while True:
        prompt = input("\nPrompt (q to quit): ")

        if prompt.lower() in {"q", "quit", "exit"}:
            break

        result = classify(prompt)

        print("\nPrediction")
        print("--------------------")
        print(f"Category   : {result['category']}")
        print(f"Difficulty : {result['difficulty']}")
        print(f"Confidence : {result['confidence']:.4f}")