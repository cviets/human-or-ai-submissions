# Yuchen: CPU CatBoost experiments

These experiments add CatBoost without modifying the team's original LightGBM pipeline.
The original 53 style features are attributed to the existing team implementation.
New work: 37 additional stylometry features, CatBoost tuning, strict cross-fitted LR
meta-features, CatBoost blending, pseudo-labeling comparisons and seed averaging.
Check the measured feature count in `run.log` rather than relying on this description.

## Run

From the repository root, using Python 3.11:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r submissions/Group_Number_19/requirements-yuchen.lock.txt
python -u submissions/Group_Number_19/yuchen_experiments.py --stage all
python submissions/Group_Number_19/verify_yuchen_results.py
python submissions/Group_Number_19/analyze_yuchen_results.py
```

Place the official `train.csv`, `unlabeled.csv`, and `test.csv` in `data/`.
By default, outputs are written to `results/yuchen/catboost_v1/`.
Alternative locations: `--data /path/to/data --out /path/to/new/run`.
Use the same output directory to resume completed stages; use a NEW directory if
data, source code, environment, thread count or iteration cap changes.
No Kaggle upload or external API classification is performed.

Stages can also be run individually:

```bash
python submissions/Group_Number_19/yuchen_experiments.py --stage style
python submissions/Group_Number_19/yuchen_experiments.py --stage stack
python submissions/Group_Number_19/yuchen_experiments.py --stage pseudo
python submissions/Group_Number_19/yuchen_experiments.py --stage stability
python submissions/Group_Number_19/yuchen_experiments.py --stage final
```

## Protocol

- Same `SEED=20261109` and 5-fold stratification as `folds.py`.
- Fold 0 is the fixed outer validation set (2,000 rows). The other 8,000 are the training pool.
- A fixed stratified 20% internal holdout divides that pool into 6,400 fitting and 1,600 tuning rows.
- CatBoost grid: depth `{4,6,8}` × L2 `{3,10}`, learning rate `0.03`, iteration cap 1,600, patience 150.
- Select hyperparameters and representation using INTERNAL AUROC. Fix the selected
  number of trees, refit on all 8,000 rows, then evaluate the 2,000 outer rows.
- Two representations: the existing style table and that table extended by
  token/character entropy, word/phrase repetition, function-word usage, case,
  contractions, punctuation patterns, and sentence/line patterns.
- LR stacking uses the team's TF-IDF + LR configuration (`C=30`), with THREE inner
  folds. Each fold fits its own vocabulary/IDF and classifier. These stricter LR
  meta-features differ from the team's original five-fold classifier-only cross-fit.
- For early stopping of the stacked CatBoost, LR cross-fitting is confined to the
  6,400 fitting rows. For the final outer fit, it is redone on the 8,000-row pool.
- CatBoost style/stack blend weights `{0,.25,.5,.75,1}` are chosen internally.
- The LightGBM comparison uses the original `fit_lgbm` function unchanged. Its LR
  feature uses the new stricter cross-fit, so it is a controlled local comparator,
  not an exact reproduction of the historical public leaderboard submission.
- LightGBM/CatBoost blends at weights `.1,.2,.3,.5` are exploratory outer-validation
  comparisons. They are not automatically chosen as final submissions.
- Pseudo-label experiments use the internally selected style CatBoost as teacher,
  confidence thresholds `.98,.95`, sample weight `.25`, one round, and the same
  hyperparameters/tree count as the supervised baseline. They use only the supplied
  unlabeled set. Deduplication checks validation text hashes only to exclude overlap;
  validation text/labels are never used for feature fitting or pseudo-label training.
- Three-seed stability holds the split, LR features and hyperparameters fixed.
- Final candidates retrain on all 10,000 labels and average three seeds. They do not
  use pseudo-labels automatically, even when an exploratory comparison improves.

The fixed validation set has already appeared in the team's previous CV work; it
is not a never-seen final test. Tiny differences near AUC 1 should not be treated
as proof of a leaderboard improvement. Public/private performance remains unknown
until an actual competition submission is evaluated.

## Outputs

- `manifest.json`: hashes of input data and relevant code; package versions; seed/config.
- `split.csv`: exact row membership in fitting/tuning/validation partitions.
- `data_audit.json`: missing/duplicate checks and normalized exact-text overlap counts.
- `*_selection.json`: complete CatBoost grid and selected configuration.
- `metrics.json`, `RESULTS.md`: measured local AUROC/log loss and runtime.
- `val_*.csv`: validation ID, ground truth and predicted probability.
- `subgroups.json`: length-bin and blank-line-group metrics; one-class AUROC is null.
- `importance_*.csv`: CatBoost feature importance.
- `cat_pseudo_*_selection.csv`: unlabeled ID, teacher probability and inclusion flag.
- `submission_*.csv`: full-data predictions; `id,label`, finite probabilities in `[0,1]`.
- `*.cbm` and `cache/`: local models and cached cross-fitting outputs; do not commit.

The repository ignores CSV/model data, and this checkout locally excludes `results/`.
Small measured summaries should be copied into the team directory for submission.

`analyze_yuchen_results.py` additionally creates paired stratified bootstrap
intervals (1,000 resamples), a validation comparison plot and up to five false
positives/negatives per main CatBoost variant for manual review. Bootstrap
intervals describe uncertainty on this validation sample; they do not capture
model-selection bias or train/test domain shift.

## Attribution / AI assistance

Codex helped implement the new experimental runner and features, download the official
competition data through the user's logged-in Kaggle session, run experiments, and
check outputs. Team-authored TF-IDF, style features and LightGBM code are reused with
attribution. This note does not claim the user has reviewed all results yet; the user
should inspect the comparisons, understand the selected model, and record their own
modeling decisions before submitting the assignment.

---

# v2: shift-aware CatBoost (Yuchen, 2026-11-10)

## Why v1 needed a different validation

v1 picked models on a random holdout where every candidate scores ~0.999, yet the
team's LightGBM got 0.9754 on the public leaderboard. The gap is distribution shift:
topic cluster 4 (poetry / fiction / book summaries) is 35.8% of test but 1.5% of train.
Within that cluster the v1 ranking is essentially "blank line => AI": all 444 blank-line
test texts in cluster 4 get P(AI)>0.5, and 630 non-blank texts are ranked by weaker
cues. Human poems in test have no blank lines (same formatting as train humans), so the
blank-line rule still holds; the loss comes from ranking the non-blank AI texts.

v2 therefore selects with **leave-one-topic-cluster-out CV (LCO)**, AUROC weighted to the
test cluster mix (the team's proxy; LightGBM style+LR scored 0.9939 here vs 0.9754 LB),
plus the same AUROC restricted to non-blank-line texts.

## What changed

1. `yuchen_features_v2.py`: 36 label-free, topic-independent features: word rarity from
   a general-English unigram model (`wordfreq` Zipf: mean/std/percentiles, rare/OOV
   share), zlib/lzma compressibility, poem structure (end-rhyme rate, line-initial case,
   line-end punctuation, enjambment, archaic words), layout-free punctuation habits.
2. Test-time self-training: CatBoost teacher (3 seeds) labels test rows with
   P>0.95 or P<0.05 (2,652 of 3,000; 1,224 as AI); the student retrains on
   train + those rows (weight 1). Uses test **text only**, never labels. In LCO this is
   simulated by self-training on the held-out cluster.
3. CatBoost depth 6, L2 3, lr 0.05, early stopping on a 10% holdout, 3-seed average;
   LR meta-feature uses the strict 3-fold cross-fit (vocabulary refit per inner fold).

Tried and dropped: augmenting train by collapsing blank lines in AI texts (LCO weighted
AUC 0.9951 -> 0.9934 for LightGBM; it erases a signal that still holds in test);
3 self-training rounds (no gain over 1 for CatBoost); LightGBM blend (+0.0002, noise).

## Results (leave-cluster-out, measured by `yuchen_v2.py --stage cv`)

| Model | LCO test-weighted AUC | non-blank weighted AUC | pooled AUC |
|---|---:|---:|---:|
| v1 features (90) + LR, CatBoost | 0.99637 | 0.99090 | 0.99486 |
| v2 features (126) + LR, CatBoost | 0.99639 | 0.99127 | 0.99487 |
| **v2 + LR + test self-training** | **0.99749** | **0.99369** | **0.99656** |

Earlier LightGBM screening (same LCO): style 53 -> 0.9851, +v1 -> 0.9899, +v2 -> 0.9918
(no LR); with LR 0.9936 (53) -> 0.9951 (126) -> 0.9967 (+ self-training).
With CatBoost the v2 features add little once the LR score is present; most of the
gain comes from self-training. LCO is still optimistic (train cluster 4 is mostly
wiki-style text, not poems), so only the leaderboard can confirm.

## Run

```bash
pip install -r submissions/Group_Number_19/requirements-yuchen.lock.txt   # now includes wordfreq
python submissions/Group_Number_19/yuchen_v2.py --stage features
python submissions/Group_Number_19/yuchen_v2.py --stage lr        # ~15 min
python submissions/Group_Number_19/yuchen_v2.py --stage predict   # writes the two submissions
python submissions/Group_Number_19/yuchen_v2.py --stage cv        # optional, LCO table above
```

Outputs in `results/yuchen/v2_shift/`: `submission_v2_cat.csv` (inductive) and
`submission_v2_cat_pseudo.csv` (with test self-training), `lco_results.csv`,
per-row LCO predictions, `predict_info.json`.

Note for the report: self-training on test text is transductive. It doesn't use
labels or external models, but confirm with course staff that it is allowed before
you pick it as a final submission.

AI assistance (v2): Claude (Cowork) diagnosed the shift, implemented v2 features and
runner, and ran the experiments; results above are copied from the run output.

