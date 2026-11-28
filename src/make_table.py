"""Build the comparison tables from the CSVs in results/.

Usage: python src/make_table.py --seed 42
Writes results/results_table.md (mine vs paper, with deltas) and
       results/per_language_table.md
"""
import argparse
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"

# (accuracy, F1) from Table 4 of the paper
PAPER = {
    ("TF-IDF + Naive Bayes", "default"): (84.5, 83.9),
    ("TF-IDF + Naive Bayes", "fewshot"): (64.6, 69.6),
    ("TF-IDF + Random Forest", "default"): (87.7, 86.9),
    ("TF-IDF + Random Forest", "fewshot"): (67.1, 57.1),
    ("XLM-RoBERTa", "default"): (94.8, 94.9),
    ("XLM-RoBERTa", "fewshot"): (76.6, 80.1),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    seed = parser.parse_args().seed

    # --- overall: mine vs paper ---
    files = [RESULTS / f"baselines_seed{seed}.csv",
             RESULTS / f"xlmr_default_seed{seed}.csv", RESULTS / f"xlmr_fewshot_seed{seed}.csv"]
    mine = pd.concat([pd.read_csv(f) for f in files if f.exists()])

    lines = ["| Model | Split | Paper acc / F1 | Mine acc / F1 | Δ acc | Δ F1 |", "|---|---|---|---|---|---|"]
    for (model, split), (paper_acc, paper_f1) in PAPER.items():
        row = mine[(mine["model"] == model) & (mine["split"] == split)]
        if row.empty:
            lines.append(f"| {model} | {split} | {paper_acc} / {paper_f1} | not run | | |")
            continue
        acc, f1 = row.iloc[0]["accuracy"], row.iloc[0]["f1"]
        lines.append(f"| {model} | {split} | {paper_acc} / {paper_f1} | {acc} / {f1} "
                     f"| {acc - paper_acc:+.1f} | {f1 - paper_f1:+.1f} |")
    (RESULTS / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), "\n")

    # --- per language: one column per model and split ---
    files = [RESULTS / f"baselines_per_language_seed{seed}.csv",
             RESULTS / f"xlmr_per_language_default_seed{seed}.csv",
             RESULTS / f"xlmr_per_language_fewshot_seed{seed}.csv"]
    per_language = pd.concat([pd.read_csv(f) for f in files if f.exists()])
    short = {"TF-IDF + Naive Bayes": "NB", "TF-IDF + Random Forest": "RF", "XLM-RoBERTa": "XLM-R"}
    per_language["column"] = per_language["model"].map(short) + " " + per_language["split"]
    table = per_language.pivot(index="language", columns="column", values="accuracy")
    table = table[sorted(table.columns, key=lambda c: (c.split()[1], c.split()[0]))]

    lines = ["| Language | " + " | ".join(table.columns) + " |", "|---" * (len(table.columns) + 1) + "|"]
    for language, values in table.iterrows():
        lines.append(f"| {language} | " + " | ".join(f"{v:.1f}" for v in values) + " |")
    (RESULTS / "per_language_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
