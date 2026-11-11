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

## v3: frozen GPT-2 token-probability features (`yuchen_v3.py`)

Added zero-shot detector signals from two frozen causal LMs, gpt2 (124M) and gpt2-medium
(355M): per-token log-prob mean/std/percentiles, entropy, Fast-DetectGPT analytic score,
single-model Binoculars-style ratio, rank shares (`yuchen_lm_features.py`, 15 per model),
plus 5 cross-model ratios. No fine-tuning and no labels from the LM. Computed on one RTX
2080 Ti on Unicorn (`gpt2_feats.sbatch`, `BS=8`, ~18 min for both models, all splits).

Individually the LM features are weak (best: `lm_logp_std` 0.837 weighted AUC; perplexity
and Fast-DetectGPT ~0.72 and ~0.52 on no-blank-line text), but CatBoost combines them well:

| Model (same LCO proxy as v2) | test-weighted AUC | non-blank weighted AUC | pooled AUC |
|---|---|---|---|
| LM features only (35) | 0.98553 | 0.98479 | 0.97777 |
| v2 + LR (rerun, this env) | 0.99617 | 0.99090 | 0.99499 |
| v2 + LM + LR | 0.99884 | 0.99703 | 0.99793 |
| **v2 + LM + LR + test self-training** | **0.99922** | **0.99804** | **0.99864** |

```bash
sbatch -p lil --gres=gpu:1 -c 4 --mem=24G gpt2_feats.sbatch   # on Unicorn, from ~/hai; BS=8 on 11 GB cards
# copy out/{gpt2,gpt2-medium} to results/yuchen/lm_features/, then:
python submissions/Group_Number_19/yuchen_v3.py --stage cv
python submissions/Group_Number_19/yuchen_v3.py --stage predict
```

Outputs in `results/yuchen/v3_lm/`: `submission_v3_cat.csv`, `submission_v3_cat_pseudo.csv`.
Note for the report: GPT-2 is an external pretrained model used only as a frozen feature
extractor (no generation, no labeling). Confirm with course staff that this is allowed.

Public LB: `submission_v3_cat_pseudo.csv` = **0.98869** (v2_cat_pseudo was 0.97880, team baseline 0.97543).

## v4: + Qwen3.5-2B Binoculars features, and learning from unlabeled data

**Binoculars features** (`yuchen_bino_features.py`, `bino_feats.sbatch`): observer
Qwen3.5-2B-Base, performer Qwen3.5-2B (instruct), same tokenizer. Per text: Binoculars
score log-PPL_obs / X-PPL(perf->obs), cross-model Fast-DetectGPT, per-model perplexity /
entropy / rank stats, instruct-vs-base log-prob gap (31 features). Frozen open-weight
models used locally as feature extractors (no commercial API, no generation, no labeling).
One RTX A6000, bf16, ~11 min for train (torch built for CUDA 12.8; cu13x wheels fail on
nodes with a 12.8 driver).

Zero-shot the Qwen Binoculars score is weak here (weighted AUC 0.689, ~0.50 on text
without blank lines), weaker than GPT-2's `logp_std` (0.837). Inside CatBoost it still helps:

| Model (LCO, 1 seed) | test-weighted AUC | non-blank weighted AUC | pooled AUC |
|---|---|---|---|
| Binoculars features only | 0.96598 | 0.97243 | 0.96394 |
| GPT-2 + Binoculars features only | 0.99216 | 0.99073 | 0.98885 |
| v3: v2 + GPT-2 + LR | 0.99884 | 0.99703 | 0.99793 |
| v4: v3 + Binoculars | 0.99906 | 0.99757 | 0.99855 |
| v3 + self-training (target) | 0.99922 | 0.99804 | 0.99864 |
| **v4 + self-training (target)** | **0.99939** | **0.99842** | **0.99903** |

**Learning from unlabeled data** (`yuchen_semi.py --stage semi`, v3 features): one round
of self-training, pseudo-labels for P>0.95 / P<0.05, pool varied:

| Pseudo-label pool | test-weighted AUC | non-blank | mean rows pseudo-labeled |
|---|---|---|---|
| none (supervised) | 0.99884 | 0.99703 | 0 |
| unlabeled.csv | 0.99897 | 0.99737 | 2851 |
| target texts (held-out cluster / test.csv) | 0.99922 | 0.99804 | 1138 |
| unlabeled.csv + target | 0.99923 | 0.99802 | 3989 |

Why unlabeled.csv helps little: it looks like train, not test. A TF-IDF classifier
trained to separate train from test texts (5-fold AUC 0.738) calls 6.8% of train, 44.6%
of test and only 11.1% of unlabeled rows "test-like". Self-training only helps where it
adds the shifted domain, which only the target texts provide.

**Error analysis** (v4 + self-training, LCO OOF, `results/yuchen/v4_bino/error_analysis_oof.csv`):
- Blank lines are almost a label in train (P(AI | blank line) = 0.995 vs 0.254 without).
  AUC is 0.99986 on texts with a blank line and 0.99722 without; 103 of the 107 errors at
  threshold 0.5 are on texts without blank lines. Error rate is highest for short texts
  without blank lines (<100 words: 6.3%) and for long ones (>300 words: 2.9%).
- 69 false negatives (AI called human) vs 38 false positives. The confident misses are
  AI text written in Wikipedia-biography / encyclopedic style (cluster 3, "born,
  university, american" -- the weakest cluster, AUC 0.99732) and dry technical
  abstracts; the confident false positives are short, enthusiastic human movie reviews.

Public LB: `submission_v4_unl_target.csv` (v4 features, self-training on unlabeled.csv +
test) = **0.99146** (v3 0.98869, v2 0.97880, team baseline 0.97543).

## Effect of training-set size (`yuchen_semi.py --stage curve`)

Each LCO training fold is subsampled (stratified) to a fraction of its ~8,750 texts; the
TF-IDF/LR is refit on the subsample; 2 repeats below 100%. Test-weighted LCO AUC
(non-blank-line subset in parentheses). Plot: `results/yuchen/semi/learning_curve.png`.

| train texts | TF-IDF + LR | style feats (CatBoost) | GPT-2 LM feats (CatBoost) | combined v3 |
|---|---|---|---|---|
| 438 (5%) | 0.782 (0.852) | 0.964 (0.912) | 0.931 (0.906) | 0.9807 (0.954) |
| 875 (10%) | 0.829 (0.896) | 0.974 (0.932) | 0.971 (0.963) | 0.9907 (0.977) |
| 2,188 (25%) | 0.870 (0.929) | 0.986 (0.964) | 0.983 (0.981) | 0.9972 (0.994) |
| 4,375 (50%) | 0.900 (0.945) | 0.990 (0.973) | 0.985 (0.984) | 0.9985 (0.996) |
| 8,750 (100%) | 0.916 (0.953) | 0.992 (0.979) | 0.986 (0.985) | 0.9988 (0.997) |

- TF-IDF + LR is the most data-hungry and the weakest under topic shift (it learns topic
  words); it is still rising at 100%.
- LM features saturate by ~25% of the data (they measure predictability, not topic), and
  on texts without blank lines they beat the style features from 10% of the data on.
- The families are complementary: the combined model with 25% of the data (0.9972) beats
  every single family trained on all of it.

