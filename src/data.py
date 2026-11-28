"""Load the MAiDE-up data and build the default and few-shot splits.

Usage: python src/data.py --seed 42
Writes data/splits/{default,fewshot}/{train,val,test}.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from seed import set_seed

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/multilingual_reviews_deception/data/all_data.csv"
SPLITS = ROOT / "data/splits"

# test fraction of each split, as in the paper (Table 3)
TEST_SIZE = {"default": 0.2, "fewshot": 0.99}


def load_data():
    df = pd.read_csv(RAW, index_col=0)
    df = df.rename(columns={"source": "label", "Review_Language": "language",
                            "City Name": "city", "Sentiment": "sentiment"})

    # Same text preprocessing as the authors' notebook: strip punctuation,
    # join "upside. downside", lowercase.
    up = df["Upside_Review"].fillna("").str.replace(r"[^\w\s]", "", regex=True).str.strip()
    down = df["Downside_Review"].fillna("").str.replace(r"[^\w\s]", "", regex=True).str.strip()
    df["text"] = (up + ". " + down).str.lower().str.strip().str.replace("\n", " ")

    df = df[["text", "label", "language", "city", "sentiment"]].reset_index(drop=True)
    df.index.name = "id"

    # The released file has 19,985 rows, not the 20,000 in the paper
    # (15 real Chinese reviews are missing).
    assert len(df) == 19985, len(df)
    assert df["label"].value_counts().to_dict() == {1: 10000, 0: 9985}
    assert df["language"].nunique() == 10
    return df


def make_split(df, test_size, seed):
    # Stratify on label AND language so every language is balanced in every
    # part (the authors stratify on label only).
    strata = df["label"].astype(str) + "_" + df["language"]
    trainval, test = train_test_split(df, test_size=test_size, stratify=strata, random_state=seed)

    # Validation = 10% of the training portion, never taken from test.
    strata = trainval["label"].astype(str) + "_" + trainval["language"]
    train, val = train_test_split(trainval, test_size=0.1, stratify=strata, random_state=seed)

    # Leakage check: no review appears in two parts, and none is lost.
    ids = [set(train.index), set(val.index), set(test.index)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(train) + len(val) + len(test) == len(df)
    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    df = load_data()
    for name, test_size in TEST_SIZE.items():
        out = SPLITS / name
        out.mkdir(parents=True, exist_ok=True)
        for part, frame in zip(["train", "val", "test"], make_split(df, test_size, args.seed)):
            frame.to_csv(out / f"{part}.csv")
            counts = frame["label"].value_counts()
            print(f"{name:8s} {part:5s} {len(frame):6d} rows  real={counts[0]}  generated={counts[1]}")
