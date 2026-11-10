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
