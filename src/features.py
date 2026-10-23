"""Per-subject features from keystroke timings.

    hold_time   = release_ts - press_ts
    flight_time = press_ts[i+1] - release_ts[i]   (negative when keys overlap)

Expects the output of load.clean() followed by load.typing_keys_only().
Sessions of the same subject are pooled into one row, so a subject is one
sample. Hold times are pooled; flight times and consecutive differences are
only taken inside a session, in press order.

The hold-time spread features come first on purpose: variability seemed to
matter more than the mean in the first look at subjects 67 and 68.

The hand split uses the usual touch-typing finger map and was fixed before any
model was fitted. The left/right difference is exploratory (it came from one PD
subject), so it is kept out of the core feature set.
"""
import numpy as np
import pandas as pd

LEFT_KEYS = set("qwertasdfgzxcvb")
RIGHT_KEYS = set("yuiophjklnm") | {"ntilde"}

CORE = ["ht_mean", "ht_median", "ht_std", "ht_iqr", "ht_skew", "ht_kurt", "ht_abs_diff"]
FLIGHT = ["ft_median", "ft_iqr"]
HAND = ["ht_left_median", "ht_right_median", "ht_hand_asym"]


def hold_times(df):
    return df["release_ts"] - df["press_ts"]


def flight_times(df):
    """Flight time per keystroke, aligned to df's index. The last key of a session is NaN."""
    df = df.sort_values(["session_id", "press_ts"])
    next_press = df.groupby("session_id")["press_ts"].shift(-1)
    return (next_press - df["release_ts"]).reindex(df.index)


def _iqr(s):
    return s.quantile(0.75) - s.quantile(0.25)


def _mean_abs_consecutive_diff(df):
    d = df.assign(hold=hold_times(df)).sort_values(["session_id", "press_ts"])
    return d.groupby("session_id")["hold"].diff().abs().mean()


def _one_subject(df):
    hold = hold_times(df)
    flight = flight_times(df).dropna()
    key = df["key"].str.lower()
    left = hold[key.isin(LEFT_KEYS)]
    right = hold[key.isin(RIGHT_KEYS)]
    return pd.Series(
        {
            "ht_mean": hold.mean(),
            "ht_median": hold.median(),
            "ht_std": hold.std(),
            "ht_iqr": _iqr(hold),
            "ht_skew": hold.skew(),
            "ht_kurt": hold.kurt(),
            "ht_abs_diff": _mean_abs_consecutive_diff(df),
            "ft_median": flight.median(),
            "ft_iqr": _iqr(flight),
            "ht_left_median": left.median(),
            "ht_right_median": right.median(),
            "ht_hand_asym": abs(left.median() - right.median()),
        }
    )


def subject_features(df):
    """One row per subject. Return (X, y, subject_ids); X is a DataFrame."""
    rows = {sid: _one_subject(g) for sid, g in df.groupby("subject_id", sort=True)}
    X = pd.DataFrame(rows).T
    y = df.drop_duplicates("subject_id").set_index("subject_id").loc[X.index, "label"].to_numpy()
    if X.isna().any().any():
        raise ValueError("NaN in subject features: " + str(X.columns[X.isna().any()].tolist()))
    return X, y, X.index.to_numpy()
