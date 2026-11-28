"""TF-IDF + Gaussian Naive Bayes / Random Forest on both splits.

Usage: python src/baselines.py --seed 42
Writes results/baselines_seed<seed>.csv and
       results/baselines_per_language_seed<seed>.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB

from seed import set_seed

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/splits"
RESULTS = ROOT / "results"


def read_split(name, part):
    # keep_default_na=False: a review whose text is the word "nan" or "null" stays text
    return pd.read_csv(SPLITS / name / f"{part}.csv", index_col="id", keep_default_na=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    overall, per_language = [], []
    for split in ["default", "fewshot"]:
        train, test = read_split(split, "train"), read_split(split, "test")

        # max_features=4500 is not in the paper; it comes from the authors' notebook.
        # Fit on train only, so no test vocabulary leaks into the features.
        tfidf = TfidfVectorizer(max_features=4500)
        X_train = tfidf.fit_transform(train["text"]).toarray()  # GaussianNB needs dense input
        X_test = tfidf.transform(test["text"]).toarray()

        models = {
            "TF-IDF + Naive Bayes": GaussianNB(),
            # the paper's Random Forest is unseeded; we fix it for reproducibility
            "TF-IDF + Random Forest": RandomForestClassifier(random_state=args.seed, n_jobs=4),
        }
        for name, model in models.items():
            model.fit(X_train, train["label"])
            pred = model.predict(X_test)

            # F1 is for the positive class, label 1 = generated (as in the authors' code)
            acc = 100 * accuracy_score(test["label"], pred)
            f1 = 100 * f1_score(test["label"], pred)
            overall.append({"model": name, "split": split, "seed": args.seed,
                            "accuracy": round(acc, 1), "f1": round(f1, 1)})
            print(f"{split:8s} {name:24s} acc={acc:.1f}  f1={f1:.1f}")

            correct = pd.Series(pred == test["label"].to_numpy(), index=test.index)
            for language, hits in correct.groupby(test["language"]):
                per_language.append({"model": name, "split": split, "seed": args.seed,
                                     "language": language, "n": len(hits),
                                     "accuracy": round(100 * hits.mean(), 1)})

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(overall).to_csv(RESULTS / f"baselines_seed{args.seed}.csv", index=False)
    pd.DataFrame(per_language).to_csv(RESULTS / f"baselines_per_language_seed{args.seed}.csv", index=False)
