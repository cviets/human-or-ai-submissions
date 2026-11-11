# Group_Number_19 — Human or AI?

**Kaggle team name:** `Group_Number_19`

Team baseline: LightGBM on 53 hand-crafted style features plus the out-of-fold score of a
TF-IDF logistic regression (word 1–3 + char 3–5 n-grams, binary tf, smoothed idf, C=30).
Public leaderboard AUROC: **0.9754**.

**Final submission (Yuchen's CatBoost line):** CatBoost on style + TF-IDF/LR + GPT-2 + Qwen3.5
features with self-training, rank-blended with a fine-tuned RoBERTa-base (weight 0.1).
Public leaderboard AUROC: **0.99162**. Pipeline, commands and all experiments: `YUCHEN_EXPERIMENTS.md`.

## Setup

Tested with Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, lightgbm 4.7.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data files (`train.csv`, `test.csv`) are not committed. Run every command from this
directory and point `DATA` at the folder that holds them (scripts default to `../../../data/`):

```bash
cd submissions/Group_Number_19
DATA=/path/to/data
```

## Predict using LightGBM model

```bash
python predict.py --train $DATA/train.csv --test $DATA/test.csv --out submission.csv
```

Fits the full pipeline on all of `train.csv` (~1 min on a laptop CPU) and writes
`id,label` with P(AI) for each test row.

## Validation of LightGBM

```bash
# LightGBM model, fixed stratified 5-fold CV (also reports logreg alone and style-only LightGBM)
python lgbm_stack.py --train $DATA/train.csv

# Same, leave-one-topic-cluster-out CV (8 k-means topic clusters)
python lgbm_stack.py --train $DATA/train.csv --test $DATA/test.csv --cv cluster

# TF-IDF baselines and hyperparameter sweeps
python baseline_tfidf.py --train $DATA/train.csv
python sweep_c.py --train $DATA/train.csv --svd-dims 0 --models logreg svm --Cs 0.3 1 3 10 30
```

Each script writes per-fold metrics and out-of-fold predictions as CSVs in this directory.

## Seeds and config

- Global seed `SEED = 20261109` (`folds.py`), used for folds, SVD, k-means, LightGBM and inner splits.
- Fold and cluster assignments are cached to `folds.csv` / `clusters.csv` on first run, so
  every experiment is scored on identical splits; delete them to rebuild from the seed.
- Leakage control: TF-IDF and models are refit inside each fold; LightGBM early-stops on a
  10% holdout of the training rows; the stacked LR score is out-of-fold within each training split.

## Files

| File | Purpose |
|---|---|
| `folds.py` | Fixed stratified 5-fold split |
| `clusters.py` | Topic clusters for leave-cluster-out CV |
| `baseline_tfidf.py` | TF-IDF features (tf/idf/n-gram options) + logistic regression or linear SVM |
| `sweep_c.py` | Sweeps C over models, tf/idf schemes, n-gram ranges, SVD |
| `style_features.py` | Layout, burstiness, punctuation and vocabulary features |
| `layout.py` | Optional whitespace/markdown normalization (`--strip-layout`) |
| `lgbm_stack.py` | CV of the LightGBM stack |
| `predict.py` | Fits the final model and writes `submission.csv` |

## Results (AUROC)

| Model | Random 5-fold CV | Leave-cluster-out (test-mix weighted) | Public LB |
|---|---|---|---|
| TF-IDF logistic regression | 0.9739 | 0.9160 | — |
| LightGBM, style features only | 0.9979 | 0.9858 | — |
| **LightGBM, style + LR score** | **0.9990** | **0.9939** | **0.9754** |
| Same, layout stripped (`--strip-layout`) | 0.9826 | — | 0.919 |

Lessons:
- In train, no human text contains a blank line while 66% of AI texts do; layout features
  exploit this. Removing them lowered the leaderboard score, so they help on test as well.
- Test is ~36% poetry versus ~1.5% in train. Wording-based TF-IDF transfers poorly to unseen
  genres (leave-cluster-out mean AUC 0.920), which random CV hides.
- CV still overestimates the leaderboard, suggesting test poetry differs from train poetry.

## AI assistance

No AI model or commercial LLM API was used to label or classify competition texts. GPT-2 and
Qwen3.5 appear only as frozen, open-weight models run locally to compute token-probability
features (see `YUCHEN_EXPERIMENTS.md`).

**Team baseline (TF-IDF, style features, LightGBM stack).** Claude Code (Claude Opus 5.5) was used to write the pipeline code, run
cross-validation experiments, and analyze results (TF/IDF variants, C sweeps, n-gram ranges,
linear SVM, style features, LightGBM stacking, train/test shift diagnostics). We reviewed the
code, chose which experiments to run, and checked results against the Kaggle public
leaderboard.

**Yuchen's CatBoost line (`yuchen_*.py`).** Codex (v1) and Claude Code (Claude Opus 5.5,
v2 onward) proposed ideas, wrote the code and ran the experiments. Yuchen chose among the proposals and suggested
alternatives (e.g. using the lab GPU cluster, Qwen3.5 instead of Qwen2.5, adding
`unlabeled.csv`, trying a fine-tuned transformer and other classifiers), and decided what to
submit to Kaggle.
