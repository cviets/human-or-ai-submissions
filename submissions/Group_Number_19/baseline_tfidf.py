"""Baseline: word + char TF-IDF -> logistic regression, scored with 5-fold CV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer,
)

from folds import N_FOLDS, SEED, make_folds


TF_SCHEMES = ("log", "log1p", "binary")


def _log1p(X):
    return X.log1p()  # sparse-safe: zeros stay zero


def make_vectorizer(tf: str, **kwargs):
    """TF-IDF vectorizer with term frequency
    log: 1 + ln(count), log1p: ln(1 + count), binary: 1 if present.
    The vocabulary is the same for all three; only the tf weighting differs.
    """
    if tf == "log":
        return TfidfVectorizer(sublinear_tf=True, **kwargs)
    if tf == "binary":
        return TfidfVectorizer(binary=True, **kwargs)
    if tf == "log1p":
        return Pipeline([
            ("count", CountVectorizer(**kwargs)),
            ("log1p", FunctionTransformer(_log1p, accept_sparse=True)),
            ("idf", TfidfTransformer()),
        ])
    raise ValueError(f"unknown tf scheme {tf!r}; expected one of {TF_SCHEMES}")


def build_features(svd_dim: int, tf: str = "binary") -> Pipeline:
    """TF-IDF (optionally followed by SVD); everything except the classifier."""
    features = FeatureUnion([
        ("word", make_vectorizer(
            tf, analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200_000,
            lowercase=True,
        )),
        # Keep case and whitespace: formatting is a strong signal in this data.
        ("char", make_vectorizer(
            tf, analyzer="char", ngram_range=(3, 5), min_df=3, max_features=300_000,
            lowercase=False,
        )),
    ])
    steps = [("tfidf", features)]
    if svd_dim > 0:
        steps += [
            ("svd", TruncatedSVD(n_components=svd_dim, random_state=SEED)),
            ("norm", Normalizer()),
        ]
    return Pipeline(steps)


def build_pipeline(svd_dim: int, C: float = 30.0, tf: str = "binary") -> Pipeline:
    return Pipeline([
        ("features", build_features(svd_dim, tf)),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("../../../data/train.csv"))
    parser.add_argument("--svd", type=int, default=0, help="TruncatedSVD dim (0 = off)")
    parser.add_argument("--C", type=float, default=30.0, help="inverse L2 strength")
    parser.add_argument("--tf", choices=TF_SCHEMES, default="binary")
    parser.add_argument("--oof-out", type=Path, default=Path("oof_tfidf.csv"))
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    fold = make_folds(train)
    y = train["label"].to_numpy()
    oof = np.zeros(len(train))

    for k in range(N_FOLDS):
        tr, va = fold != k, fold == k
        # Vectorizers are refit inside each fold so validation text never leaks.
        model = build_pipeline(args.svd, args.C, args.tf)
        model.fit(train.loc[tr, "text"], y[tr])
        oof[va] = model.predict_proba(train.loc[va, "text"])[:, 1]
        print(f"fold {k}: auc={roc_auc_score(y[va], oof[va]):.4f} "
              f"logloss={log_loss(y[va], oof[va]):.4f} "
              f"acc={accuracy_score(y[va], oof[va] > 0.5):.4f}")

    print(f"OOF   : auc={roc_auc_score(y, oof):.4f} "
          f"logloss={log_loss(y, oof):.4f} "
          f"acc={accuracy_score(y, oof > 0.5):.4f}")
    pd.DataFrame({"id": train["id"], "fold": fold, "label": y, "pred": oof}).to_csv(
        args.oof_out, index=False)


if __name__ == "__main__":
    main()
