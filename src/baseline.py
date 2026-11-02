"""neuroQWERTY baseline: how well do hold-time features separate PD from controls?

Run from the repo root:  python -m src.baseline

Everything uses one row per subject (85 subjects, 42 PD) and PD as the positive
class. Nothing is tuned, so there is no inner loop to leak from. Results go to
results/baseline_neuroqwerty.csv and results/baseline_roc.png.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import bootstrap_auc_ci, bootstrap_auc_diff_ci, loso_scores, train_test_auc
from src.features import CORE, FLIGHT, HAND, subject_features
from src.load import clean, load_neuroqwerty, load_subject_info, typing_keys_only

RAW = "data/raw/neuroqwerty"
OUT = "results"

# Direction is fixed beforehand: higher nQi means more PD, higher tapping counts
# and typing speed mean healthier. Not chosen by looking at the AUC.
# UPDRS is left out: it separates the classes perfectly (controls <= 6, PD >= 7),
# so it defines the label and is not an independent comparison.
REFERENCE = {"nqScore": 1, "afTap": -1, "sTap": -1, "typingSpeed": -1}


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0)


def permutation_null(X, y, ids, n_perm=200, seed=0):
    """LOSO AUC with the labels shuffled. Should centre on 0.5 if nothing leaks."""
    rng = np.random.default_rng(seed)
    return np.array([roc_auc_score(p, loso_scores(X, p, ids, logreg()))
                     for p in (rng.permutation(y) for _ in range(n_perm))])


def row(analysis, detail, y, scores):
    lo, hi = bootstrap_auc_ci(y, scores)
    return {"analysis": analysis, "detail": detail, "n": len(y),
            "auc": roc_auc_score(y, scores), "ci_low": lo, "ci_high": hi}


def main():
    df = load_neuroqwerty(RAW)
    cleaned, n_dropped, pct = clean(df)
    typed = typing_keys_only(cleaned)
    print(f"{len(df)} keystrokes -> {n_dropped} dropped by sanity rules ({pct:.2f}%), "
          f"{len(cleaned) - len(typed)} more removed as non-typing keys, {len(typed)} kept")

    X, y, ids = subject_features(typed)
    info = load_subject_info(RAW).set_index("subject_id").loc[ids]
    study = info["study"].to_numpy()
    rows = []

    updrs = info["updrs108"].to_numpy()
    assert updrs[y == 0].max() < updrs[y == 1].min(), "UPDRS no longer separates the labels"

    for col, sign in REFERENCE.items():
        ok = info[col].notna().to_numpy()
        rows.append(row("reference score", col, y[ok], sign * info[col].to_numpy()[ok]))

    for col in CORE + FLIGHT + HAND:
        rows.append(row("single feature", col, y, X[col].to_numpy()))

    sets = {"hold core": CORE, "hold core + flight": CORE + FLIGHT,
            "hold core + flight + hand": CORE + FLIGHT + HAND}
    oof = {}
    for name, cols in sets.items():
        oof[name] = loso_scores(X[cols], y, ids, logreg())
        rows.append(row("LOSO logistic regression", name, y, oof[name]))
    oof["ht_mean only"] = loso_scores(X[["ht_mean"]], y, ids, logreg())
    rows.append(row("LOSO logistic regression", "ht_mean only", y, oof["ht_mean only"]))
    rows.append(row("LOSO random forest", "hold core + flight", y,
                    loso_scores(X[CORE + FLIGHT], y, ids, forest())))

    # same subjects, same resamples: is hold time better than alternating finger tapping?
    has_tap = info["afTap"].notna().to_numpy()
    tap = -info["afTap"].to_numpy()[has_tap]
    for name in ["hold core", "hold core + flight"]:
        diff = roc_auc_score(y[has_tap], oof[name][has_tap]) - roc_auc_score(y[has_tap], tap)
        lo, hi = bootstrap_auc_diff_ci(y[has_tap], oof[name][has_tap], tap)
        rows.append({"analysis": "paired AUC difference vs afTap", "detail": name, "n": int(has_tap.sum()),
                     "auc": diff, "ci_low": lo, "ci_high": hi})

    null = permutation_null(X[CORE], y, ids)
    observed = roc_auc_score(y, oof["hold core"])
    p_value = (1 + (null >= observed).sum()) / (1 + len(null))
    rows.append({"analysis": "label permutation null", "detail": f"hold core, {len(null)} shuffles, p={p_value:.3f}",
                 "n": len(y), "auc": null.mean(), "ci_low": np.percentile(null, 2.5),
                 "ci_high": np.percentile(null, 97.5)})

    cols = CORE + FLIGHT
    for s in ["CS1PD", "CS2PD"]:
        m = study == s
        rows.append(row("LOSO within one study", s, y[m], loso_scores(X[cols][m], y[m], ids[m], logreg())))
    for train, test in [("CS1PD", "CS2PD"), ("CS2PD", "CS1PD")]:
        tr, te = study == train, study == test
        fitted = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[cols][tr], y[tr])
        scores = fitted.decision_function(X[cols][te])
        assert abs(train_test_auc(X[cols][tr], y[tr], X[cols][te], y[te], logreg()) - roc_auc_score(y[te], scores)) < 1e-9
        rows.append(row("train on one study, test on other", f"{train} -> {test}", y[te], scores))

    table = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    table.round(4).to_csv(os.path.join(OUT, "baseline_neuroqwerty.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table.round(3).to_string(index=False))

    plt.figure(figsize=(5, 5))
    for name in ["ht_mean only", "hold core", "hold core + flight"]:
        fpr, tpr, _ = roc_curve(y, oof[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, oof[name]):.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("neuroQWERTY, leave-one-subject-out")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "baseline_roc.png"), dpi=150)


if __name__ == "__main__":
    main()
