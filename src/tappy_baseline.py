"""Clinic vs home: does the neuroQWERTY hold-time result carry over to Tappy?

Run from the repo root:  python -m src.tappy_baseline

Decisions made before any Tappy model was run (I had looked at per-class median
hold times, keystroke counts and the clock grid, but at no AUC):
  - features: hold-time distribution only (HT_DIST). Tappy has no key names, and
    consecutive-key features are not comparable across Tappy's long pauses.
  - a Tappy user needs at least N_MIN clean keystrokes, where N_MIN is the
    smallest neuroQWERTY subject. A rerun at the neuroQWERTY median checks this.
  - same cleaning rules as neuroQWERTY, same model (standardised logistic
    regression), nothing tuned.
  - PD is the positive class; AUC is computed over pooled leave-one-subject-out
    scores, 95% intervals come from resampling subjects.
Everything is reported, including checks that show nothing.

The last block (marked EXPLORATORY) was added after the first run showed no signal
on Tappy: it asks whether richer features (hold time by hand, latency by key
direction) bring it back. It was not part of the plan above, so treat it as a
follow-up and not as a pre-planned test.

Writes results/clinic_vs_home.csv, results/clinic_vs_home_descriptives.csv and
results/clinic_vs_home.png.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import bootstrap_auc_ci, fit_scores, loso_scores
from src.features import HT_DIST, hold_distribution_features, tappy_features
from src.load import clean, load_neuroqwerty, load_tappy, load_tappy_users, typing_keys_only

NQ_RAW = "data/raw/neuroqwerty"
TAPPY_RAW = "data/raw/tappy"
OUT = "results"
CLOCK_TICK = 1 / 128  # seconds; the grid most Tappy hold times sit on
MATCHED_N = 1500  # about the neuroQWERTY median keystrokes per subject
N_REPEATS = 5
N_PHASES = 20
N_PERM = 100


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def row(analysis, detail, y, scores):
    lo, hi = bootstrap_auc_ci(y, scores)
    return {"analysis": analysis, "detail": detail, "n": len(y),
            "auc": roc_auc_score(y, scores), "ci_low": lo, "ci_high": hi}


def users_with_enough(df, min_n=30):
    """Users with at least min_n keystrokes on every hand and every letter-to-letter direction."""
    by_hand = df.groupby(["subject_id", "hand"], observed=True).size().unstack(fill_value=0)
    by_dir = df.groupby(["subject_id", "direction"], observed=True).size().unstack(fill_value=0)[["LL", "LR", "RL", "RR"]]
    ok = (by_hand >= min_n).all(axis=1) & (by_dir >= min_n).all(axis=1)
    return ok[ok].index


def forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0)


def permutation_row(name, X, y, ids, oof, rng):
    """LOSO AUC with shuffled labels. Leave-one-out sits below 0.5 when there is no signal."""
    null = np.array([roc_auc_score(p, loso_scores(X, p, ids, logreg()))
                     for p in (rng.permutation(y) for _ in range(N_PERM))])
    p_value = (1 + (null >= roc_auc_score(y, oof)).sum()) / (1 + N_PERM)
    return {"analysis": "label permutation null", "detail": f"{name}, {N_PERM} shuffles, p={p_value:.3f}",
            "n": len(y), "auc": null.mean(), "ci_low": np.percentile(null, 2.5), "ci_high": np.percentile(null, 97.5)}


def spread_row(analysis, detail, n, aucs):
    """Mean AUC over repeated random draws; the interval is the min to max, not a CI."""
    return {"analysis": analysis, "detail": detail, "n": n, "auc": float(np.mean(aucs)),
            "ci_low": float(np.min(aucs)), "ci_high": float(np.max(aucs))}


def cohens_d(x, y):
    a, b = x[y == 1], x[y == 0]
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def quantize(df, step, rng):
    """Mimic a recorder whose clock ticks every `step` seconds, with a random phase per session."""
    phase = pd.Series(rng.uniform(0, step, df["session_id"].nunique()), index=df["session_id"].unique())
    off = df["session_id"].map(phase).to_numpy()
    out = df.copy()
    out["press_ts"] = np.floor((df["press_ts"].to_numpy() + off) / step) * step
    out["release_ts"] = np.floor((df["release_ts"].to_numpy() + off) / step) * step
    return out


def describe(name, X, y, kept_counts):
    ms = X[["ht_median", "ht_std"]] * 1000
    return {"dataset": name, "subjects": len(y), "pd": int(y.sum()), "share_pd": round(y.mean(), 3),
            "median_keystrokes": int(kept_counts.median()),
            "median_hold_ms_control": round(ms["ht_median"][y == 0].median(), 1),
            "median_hold_ms_pd": round(ms["ht_median"][y == 1].median(), 1),
            "std_hold_ms_control": round(ms["ht_std"][y == 0].median(), 1),
            "std_hold_ms_pd": round(ms["ht_std"][y == 1].median(), 1),
            "cohens_d_median": round(cohens_d(X["ht_median"].to_numpy(), y), 2),
            "cohens_d_std": round(cohens_d(X["ht_std"].to_numpy(), y), 2)}


def plot_summary(table):
    shown = [("LOSO logistic regression", "neuroQWERTY", "neuroQWERTY, LOSO (clinic)"),
             ("LOSO logistic regression", "Tappy", "Tappy, LOSO (home)"),
             ("Tappy, matched to neuroQWERTY data size", None, "Tappy, data size matched"),
             ("transfer", "train neuroQWERTY, test Tappy", "train clinic, test home"),
             ("transfer", "train Tappy, test neuroQWERTY", "train home, test clinic"),
             ("neuroQWERTY with a coarser clock", "7.8 ms", "clinic, 7.8 ms clock"),
             ("neuroQWERTY with a coarser clock", "15.6 ms", "clinic, 15.6 ms clock")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (analysis, key, label) in enumerate(reversed(shown)):
        r = table[(table["analysis"] == analysis) & (key is None or table["detail"].str.contains(key, regex=False))].iloc[0]
        ax.errorbar(r["auc"], i, xerr=[[r["auc"] - r["ci_low"]], [r["ci_high"] - r["auc"]]], fmt="o", color="C0", capsize=3)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([label for _, _, label in reversed(shown)])
    ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUC (bars: 95% CI, or range for repeated draws)")
    ax.set_title("Hold-time distribution features, clinic vs home")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "clinic_vs_home.png"), dpi=150)


def main():
    nq_typed = typing_keys_only(clean(load_neuroqwerty(NQ_RAW))[0])
    nq_counts = nq_typed.groupby("subject_id").size()
    n_min = int(nq_counts.min())
    Xn, yn, idn = hold_distribution_features(nq_typed)

    tp_raw = load_tappy(TAPPY_RAW)
    tp_clean, n_drop, pct = clean(tp_raw)
    print(f"Tappy: {len(tp_raw)} keystrokes, {tp_raw.attrs['n_malformed']} malformed rows dropped, "
          f"{n_drop} removed by the sanity rules ({pct:.2f}%)")
    tp_counts = tp_clean.groupby("subject_id").size()
    keep = tp_counts[tp_counts >= n_min].index
    tp = tp_clean[tp_clean["subject_id"].isin(keep)]
    Xt, yt, idt = hold_distribution_features(tp)
    print(f"Tappy users with a label and data: {len(tp_counts)}; with >= {n_min} keystrokes: {len(keep)} "
          f"({int(yt.sum())} PD, {int((1 - yt).sum())} control)")

    rows = []
    desc = pd.DataFrame([describe("neuroQWERTY", Xn, yn, nq_counts),
                         describe("Tappy", Xt, yt, tp_counts[keep])])
    print(desc.T.to_string(header=False))

    for name, X, y in [("neuroQWERTY", Xn, yn), ("Tappy", Xt, yt)]:
        for col in HT_DIST:
            rows.append(row(f"single feature, {name}", col, y, X[col].to_numpy()))

    oof_n = loso_scores(Xn, yn, idn, logreg())
    oof_t = loso_scores(Xt, yt, idt, logreg())
    rows.append(row("LOSO logistic regression", "neuroQWERTY", yn, oof_n))
    rows.append(row("LOSO logistic regression", "Tappy", yt, oof_t))

    rows.append(row("transfer", "train neuroQWERTY, test Tappy", yt, fit_scores(Xn, yn, Xt, logreg())))
    rows.append(row("transfer", "train Tappy, test neuroQWERTY", yn, fit_scores(Xt, yt, Xn, logreg())))

    big = tp_counts[tp_counts >= MATCHED_N].index
    tp_big = tp_clean[tp_clean["subject_id"].isin(big)]
    Xb, yb, idb = hold_distribution_features(tp_big)
    rows.append(row("Tappy, only users with >= 1500 keystrokes", "all their keystrokes", yb,
                    loso_scores(Xb, yb, idb, logreg())))
    aucs = []
    for rep in range(N_REPEATS):
        sample = tp_big.groupby("subject_id", group_keys=False).sample(n=MATCHED_N, random_state=rep)
        Xs, ys, ids_ = hold_distribution_features(sample)
        aucs.append(roc_auc_score(ys, loso_scores(Xs, ys, ids_, logreg())))
    rows.append(spread_row("Tappy, matched to neuroQWERTY data size",
                           f"{MATCHED_N} random keystrokes per user, {N_REPEATS} draws, range", len(yb), aucs))

    rng = np.random.default_rng(0)
    for step, label in [(CLOCK_TICK, "7.8 ms"), (2 * CLOCK_TICK, "15.6 ms")]:
        aucs = []
        for _ in range(N_PHASES):
            Xq, yq, idq = hold_distribution_features(quantize(nq_typed, step, rng))
            aucs.append(roc_auc_score(yq, loso_scores(Xq, yq, idq, logreg())))
        rows.append(spread_row("neuroQWERTY with a coarser clock", f"{label} ticks, {N_PHASES} random phases, range",
                               len(yn), aucs))

    users = load_tappy_users(TAPPY_RAW).set_index("subject_id").loc[idt]
    for flag, name in [("True", "PD on levodopa"), ("False", "PD not on levodopa")]:
        m = ((users["label"] == 0) | ((users["label"] == 1) & (users["Levadopa"] == flag))).to_numpy()
        rows.append(row("Tappy subgroup vs controls", name, yt[m], oof_t[m]))
    for level in ["Mild", "Medium", "Severe"]:
        m = ((users["label"] == 0) | ((users["label"] == 1) & (users["Impact"] == level))).to_numpy()
        rows.append(row("Tappy subgroup vs controls", f"PD, self-reported impact {level}", yt[m], oof_t[m]))

    birth = pd.to_numeric(users["BirthYear"], errors="coerce").to_numpy()
    has = ~np.isnan(birth)
    rows.append(row("Tappy age check", "birth year alone (older = more PD)", yt[has], -birth[has]))

    rng = np.random.default_rng(0)
    rows.append(permutation_row("Tappy, hold-time distribution features", Xt, yt, idt, oof_t, rng))

    # Added after the result above: Tappy also records latency and hand/direction, so check
    # whether richer features bring the signal back before concluding it is gone.
    keep_r = users_with_enough(tp)
    Xr, yr, idr = tappy_features(tp[tp["subject_id"].isin(keep_r)])
    print(f"Richer features need >= 30 keystrokes per hand and direction: {len(yr)} of {len(yt)} users kept")
    rows.append(row("EXPLORATORY Tappy richer features", f"hold-time distribution only, same {len(yr)} users", yr,
                    loso_scores(Xr[HT_DIST], yr, idr, logreg())))
    oof_r = loso_scores(Xr, yr, idr, logreg())
    for col in Xr.columns.difference(HT_DIST):
        rows.append(row("EXPLORATORY single feature, Tappy", col, yr, Xr[col].to_numpy()))
    rows.append(row("EXPLORATORY Tappy richer features", f"LOSO logistic regression, {Xr.shape[1]} features", yr, oof_r))
    rows.append(row("EXPLORATORY Tappy richer features", f"LOSO random forest, {Xr.shape[1]} features", yr,
                    loso_scores(Xr, yr, idr, forest())))
    rows.append(permutation_row("Tappy, richer features", Xr, yr, idr, oof_r, rng))

    table = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    table.round(4).to_csv(os.path.join(OUT, "clinic_vs_home.csv"), index=False)
    desc.to_csv(os.path.join(OUT, "clinic_vs_home_descriptives.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 70):
        print(table.round(3).to_string(index=False))

    plot_summary(table)


if __name__ == "__main__":
    main()
