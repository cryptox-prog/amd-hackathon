#!/usr/bin/env python3

import json
from pathlib import Path

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

CHECKPOINT = Path("checkpoints/router")

with open(CHECKPOINT / "label_map.json") as f:
    label_map = json.load(f)

ID2LABEL = {int(k): v for k, v in label_map["id2label"].items()}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

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

    category = ID2LABEL[prediction.item()]

    return {
        "category": category,
        "difficulty": "hard",      # TODO: Replace with difficulty classifier
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
        print("Category   :", result["category"])
        print("Difficulty :", result["difficulty"])
        print("Confidence :", result["confidence"])