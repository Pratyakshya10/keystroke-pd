"""Per-subject feature extraction from keystroke timings.

Start with the smallest possible set. Add features only after the baseline
runs end to end.

    hold_time   = release_ts - press_ts
    flight_time = press_ts[i+1] - release_ts[i]

Per subject, aggregate: mean, std, skew, kurtosis, and the mean absolute
consecutive difference of hold time.

Variability matters more than the mean. Keep that in mind when choosing
aggregates.
"""


def hold_times(df):
    raise NotImplementedError


def flight_times(df):
    raise NotImplementedError


def subject_features(df):
    """One row per subject. Return (X, y, subject_ids)."""
    raise NotImplementedError
