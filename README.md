# MAiDE-up reproduction (CSCI 3052U)

Reproduction of the deception-detection experiments in
[MAiDE-up: Multilingual Deception Detection of AI-generated Hotel Reviews](https://aclanthology.org/2025.findings-naacl.88.pdf)
(Ignat, Xu and Mihalcea, NAACL Findings 2025), plus a per-language breakdown of every model.

Role: Hacker and Implementer for Reproducibility.

Data and the reference notebook come from the authors' repository,
[MichiganNLP/multilingual_reviews_deception](https://github.com/MichiganNLP/multilingual_reviews_deception).
The data is not copied into this repo. The code in `src/` is my own; settings taken from the
authors' notebook are listed under "Settings not in the paper".

## Pipeline

```
all_data.csv  ->  preprocess  ->  split (seed 42)  ->  models                   ->  evaluate
19,985 reviews    join upside     default 80/20        TF-IDF + Naive Bayes         accuracy, F1
10 languages      + downside,     few-shot 1/99        TF-IDF + Random Forest       overall and
real vs GPT-4     clean           10% of train = val   XLM-RoBERTa (fine-tuned)     per language
```

## Structure

```
src/seed.py        set_seed(): seeds random, numpy and torch
src/data.py        loads the data, builds and saves both splits, checks for leakage
src/baselines.py   TF-IDF + Gaussian Naive Bayes / Random Forest, both splits
src/xlmr.py        fine-tunes XLM-RoBERTa base on one split
src/make_table.py  builds the mine-vs-paper and per-language tables from results/
data/splits/       the saved train / val / test files used by every model
results/           one CSV per run: overall and per-language
presentation/      slides
requirements.txt   pinned library versions
```

## Steps to reproduce

Python 3.14. Every script takes `--seed` (default 42).

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data (authors' repo, not committed here)
git clone https://github.com/MichiganNLP/multilingual_reviews_deception data/raw/multilingual_reviews_deception

# 3. Splits -> data/splits/{default,fewshot}/{train,val,test}.csv
#    (already committed; rerunning with seed 42 recreates them)
python src/data.py --seed 42

# 4. Baseline rows (TF-IDF + Naive Bayes / Random Forest, both splits), about 40 s on CPU
python src/baselines.py --seed 42

# 5. XLM-RoBERTa few-shot row, about 1 min on a 16 GB GPU
python src/xlmr.py --split fewshot --seed 42

# 6. XLM-RoBERTa default row, about 10 min on a 16 GB GPU
python src/xlmr.py --split default --seed 42

# 7. Comparison tables -> results/results_table.md, results/per_language_table.md
python src/make_table.py --seed 42
```

Each step prints its accuracy and F1 and writes them to `results/`.

## Results (seed 42)

Accuracy / F1 in percent. F1 is for the generated class.

| Model | Split | Paper acc / F1 | Mine acc / F1 | Δ acc | Δ F1 |
|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | default | 84.5 / 83.9 | 84.4 / 83.7 | -0.1 | -0.2 |
| TF-IDF + Naive Bayes | fewshot | 64.6 / 69.6 | 71.0 / 73.7 | +6.4 | +4.1 |
| TF-IDF + Random Forest | default | 87.7 / 86.9 | 86.6 / 85.9 | -1.1 | -1.0 |
| TF-IDF + Random Forest | fewshot | 67.1 / 57.1 | 69.6 / 59.8 | +2.5 | +2.7 |
| XLM-RoBERTa | default | 94.8 / 94.9 | 96.7 / 96.7 | +1.9 | +1.8 |
| XLM-RoBERTa | fewshot | 76.6 / 80.1 | 81.4 / 82.0 | +4.8 | +1.9 |

## Per-language accuracy

The paper reports per-language results only for few-shot XLM-RoBERTa, and only as a chart.

| Language | NB default | RF default | XLM-R default | NB fewshot | RF fewshot | XLM-R fewshot |
|---|---|---|---|---|---|---|
| Chinese | 49.9 | 50.1 | 95.7 | 49.5 | 49.6 | 80.1 |
| English | 87.2 | 87.5 | 97.5 | 73.9 | 76.0 | 77.8 |
| French | 91.2 | 91.5 | 96.8 | 72.0 | 78.8 | 83.0 |
| German | 85.8 | 89.0 | 96.5 | 72.7 | 80.6 | 83.9 |
| Italian | 87.0 | 91.5 | 97.0 | 77.8 | 76.5 | 79.0 |
| Korean | 88.0 | 88.5 | 95.8 | 65.3 | 55.4 | 76.1 |
| Romanian | 86.8 | 90.5 | 95.5 | 76.1 | 76.1 | 86.1 |
| Russian | 88.2 | 91.2 | 97.5 | 71.5 | 56.7 | 77.5 |
| Spanish | 90.5 | 94.8 | 98.5 | 73.9 | 67.3 | 83.6 |
| Turkish | 89.5 | 91.2 | 96.5 | 77.4 | 78.5 | 87.2 |

Findings:

- **TF-IDF baselines are at chance on Chinese.** Chinese is written without spaces and the
  preprocessing strips punctuation, so scikit-learn's word tokenizer turns each review into one
  long token. 96% of Chinese training reviews end up with no TF-IDF features at all. The overall
  baseline numbers hide this.
- **XLM-RoBERTa does not have this problem**, because its subword tokenizer does not rely on spaces.
- **Few-shot XLM-RoBERTa matches the paper's ranking**: Korean and English are among the hardest
  languages, Romanian and German among the easiest.

## Settings not in the paper

| Setting | Value | Source |
|---|---|---|
| Random seed | 42 | authors' notebook |
| TF-IDF vocabulary | `max_features=4500` | authors' notebook |
| Text preprocessing | strip punctuation, join "upside. downside", lowercase | authors' notebook |
| F1 | binary, positive class = generated | authors' notebook |
| Weight decay, fp16 | 0.01, on | authors' notebook |
| Validation | 10% of the training portion (so the model trains on about 7,200 + 7,200) | authors' notebook |

## Differences from the authors' setup

- **Stratification**: splits are stratified on label and language together (the authors stratify on
  label only), so every language is balanced in train, validation and test.
- **Random Forest is seeded** (`random_state=seed`); the authors' is not.
- **Best epoch** is chosen by validation accuracy, as the paper states. The authors' notebook uses
  the Trainer default, which is validation loss.
- **Padding**: batches are padded to their longest review, not to 512 tokens. This only affects speed.
- **Dataset size**: the released file has 19,985 reviews, not 20,000 (15 real Chinese reviews are
  missing), so split sizes are 14,389 / 1,599 / 3,997 (default) and 179 / 20 / 19,786 (few-shot).

## Limitations

- Results are from a single seed. The few-shot split trains on 179 reviews and validates on 20, so
  its numbers are sensitive to the seed.
- GPU results can vary slightly across hardware and library versions even with fixed seeds.
