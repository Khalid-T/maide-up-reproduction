"""Fine-tune XLM-RoBERTa base on one split.

Usage: python src/xlmr.py --split fewshot --seed 42
Writes results/xlmr_<split>_seed<seed>.csv and
       results/xlmr_per_language_<split>_seed<seed>.csv
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from seed import set_seed

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/splits"
RESULTS = ROOT / "results"
MODEL = "xlm-roberta-base"


class ReviewDataset(torch.utils.data.Dataset):
    """Tokenized reviews plus their labels, one item per review."""

    def __init__(self, frame, tokenizer):
        # truncation=True cuts reviews longer than the model's 512-token limit
        self.encodings = tokenizer(frame["text"].tolist(), truncation=True)
        self.labels = frame["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: values[i] for key, values in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


def accuracy_metric(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": accuracy_score(labels, np.argmax(logits, axis=-1))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["default", "fewshot"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    def read(part):
        return pd.read_csv(SPLITS / args.split / f"{part}.csv", index_col="id", keep_default_na=False)

    train, val, test = read("train"), read("val"), read("test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    checkpoints = ROOT / "checkpoints" / f"{args.split}_seed{args.seed}"
    training_args = TrainingArguments(
        output_dir=str(checkpoints),
        # hyperparameters from the paper (weight decay and fp16 from the authors' notebook)
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # half precision only exists on GPU
        per_device_eval_batch_size=64,
        # evaluate on validation after every epoch and keep the best epoch by accuracy
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ReviewDataset(train, tokenizer),
        eval_dataset=ReviewDataset(val, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),  # pad each batch to its longest review
        compute_metrics=accuracy_metric,
    )
    trainer.train()

    # The test set is only touched here, after the best epoch has been chosen.
    logits = trainer.predict(ReviewDataset(test, tokenizer)).predictions
    pred = np.argmax(logits, axis=-1)

    acc = 100 * accuracy_score(test["label"], pred)
    f1 = 100 * f1_score(test["label"], pred)  # positive class = 1 = generated
    print(f"{args.split} XLM-RoBERTa acc={acc:.1f} f1={f1:.1f}")

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame([{"model": "XLM-RoBERTa", "split": args.split, "seed": args.seed,
                   "accuracy": round(acc, 1), "f1": round(f1, 1)}]).to_csv(
        RESULTS / f"xlmr_{args.split}_seed{args.seed}.csv", index=False)

    correct = pd.Series(pred == test["label"].to_numpy(), index=test.index)
    rows = [{"model": "XLM-RoBERTa", "split": args.split, "seed": args.seed, "language": language,
             "n": len(hits), "accuracy": round(100 * hits.mean(), 1)}
            for language, hits in correct.groupby(test["language"])]
    pd.DataFrame(rows).to_csv(RESULTS / f"xlmr_per_language_{args.split}_seed{args.seed}.csv", index=False)

    shutil.rmtree(checkpoints)  # free disk; the results are what we keep
