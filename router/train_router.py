#!/usr/bin/env python3
"""
train_router.py

A production-ready script to fine-tune DistilBERT as an 8-class text classifier
for routing user prompts in an AMD Developer Hackathon agent.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Scikit-learn metrics and splitting
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    data_path: Path = Path("data/synthetic_dataset.json")
    output_dir: Path = Path("checkpoints/router")
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 256
    seed: int = 42
    test_size: float = 0.2
    num_labels: int = 16


# Defined fixed mapping for categories
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
LABEL2ID = {label: idx for idx, label in enumerate(CLASSES)}
ID2LABEL = {idx: label for idx, label in enumerate(CLASSES)}


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility across platforms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in standard algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Automatically detect and return the target computing device hardware."""
    if torch.cuda.is_available():
        logger.info("Target Device: NVIDIA CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("Target Device: Apple Silicon MPS")
        return torch.device("mps")
    else:
        logger.info("Target Device: CPU")
        return torch.device("cpu")


def load_dataset(data_path: Path) -> Tuple[List[str], List[int]]:
    """Load JSON raw data and filter down to inputs and mapped integer tags."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file missing at destination: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    texts: List[str] = []
    labels: List[int] = []

    for item in raw_data:
        prompt = item.get("prompt")
        category = item.get("category")
        difficulty = item.get("difficulty")

        label = f"{category}_{difficulty}"

        if prompt and label in LABEL2ID:
            texts.append(prompt)
            labels.append(LABEL2ID[label])

    logger.info(f"Successfully processed {len(texts)} valid samples from source.")
    return texts, labels


class RouterDataset(Dataset):
    """Custom PyTorch dataset wrapper for tokenizing textual inputs on-the-fly."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: DistilBertTokenizerFast,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove the batch dimension added by return_tensors="pt"
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_epoch(
    model: DistilBertForSequenceClassification,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Execute a single training epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: DistilBertForSequenceClassification,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate performance, return validation loss, accuracy, and raw predictions."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = correct_predictions / total_samples
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy, all_labels, all_preds


def main() -> None:
    config = TrainingConfig()
    set_seed(config.seed)
    device = get_device()

    # Load and Split Data (Stratified)
    texts, labels = load_dataset(config.data_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        stratify=labels,
        random_state=config.seed,
    )

    # Initialize Tokenizer and Datasets
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)

    train_dataset = RouterDataset(
        train_texts, train_labels, tokenizer, config.max_length
    )
    val_dataset = RouterDataset(val_texts, val_labels, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize Architecture
    model = DistilBertForSequenceClassification.from_pretrained(
    config.model_name,
    num_labels=config.num_labels,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info("Starting model fine-tuning initialization...")
    best_acc = 0.0
    # Training Loop
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{config.epochs} -> "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f}"
        )
        if val_acc > best_acc:
                best_acc = val_acc
                model.save_pretrained(config.output_dir)
                tokenizer.save_pretrained(config.output_dir)
                print(f"✓ Saved new best model (Val Acc = {best_acc:.4f})")

    model = DistilBertForSequenceClassification.from_pretrained(
    config.output_dir
    )
    model.to(device)
    # Final Evaluation for Global Metrics
    _, _, final_labels, final_preds = evaluate(model, val_loader, criterion, device)

    # Compute Explicit Target Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        final_labels, final_preds, average="macro"
    )

    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE EVALUATION METRICS")
    print("=" * 60)
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1 Score  : {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(final_labels, final_preds, target_names=CLASSES))

    print("Confusion Matrix:")
    print(confusion_matrix(final_labels, final_preds))
    print("=" * 60)
    
    # Build internal structured configuration label map mapping schema
    # JSON standard keys must be string objects
    label_map_payload = {
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }

    label_map_path = config.output_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as json_file:
        json.dump(label_map_payload, json_file, indent=4)

    logger.info(f"Production components successfully written to {config.output_dir}")


if __name__ == "__main__":
    main()