"""Load the neuroQWERTY (MIT-CSXPD) typing sessions into one DataFrame.

Raw session files have no header and four columns: key, hold, release, press.
Times are seconds from the start of the session. Release comes *before* press
in the file. Checked against release - press == hold and against nqDataLoader.py
(f2 is release, f3 is press).

Output of load_neuroqwerty(), one row per keystroke:
    subject_id : str    e.g. "CS1PD-68" (study + pID)
    study      : str    "CS1PD" or "CS2PD"
    session_id : str    raw file name without .csv
    key        : str
    press_ts   : float  seconds
    release_ts : float  seconds
    label      : int    1 = PD, 0 = control

Rows keep the order they have in the file, which the cleaning step relies on.
"""
import os

import numpy as np
import pandas as pd

STUDIES = ["CS1PD", "CS2PD"]
RAW_COLUMNS = ["key", "hold", "release", "press"]


def load_subject_info(raw_dir):
    """One row per subject from the GT_DataPD files (label, UPDRS, tapping, ...)."""
    tables = []
    for study in STUDIES:
        path = os.path.join(raw_dir, f"MIT-{study}", f"GT_DataPD_MIT-{study}.csv")
        info = pd.read_csv(path)
        info["study"] = study
        info["subject_id"] = study + "-" + info["pID"].astype(str)
        info["label"] = info["gt"].astype(int)
        tables.append(info)
    return pd.concat(tables, ignore_index=True)


def _read_session(path):
    # keep_default_na=False: a key literally named "NA" or "None" must stay a string
    return pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        dtype={"key": str, "hold": float, "release": float, "press": float},
        keep_default_na=False,
    )


def load_neuroqwerty(raw_dir):
    """Return every session of both studies in the target schema."""
    info = load_subject_info(raw_dir)
    frames = []
    for _, subject in info.iterrows():
        study_dir = os.path.join(raw_dir, f"MIT-{subject['study']}", f"data_MIT-{subject['study']}")
        for col in [c for c in info.columns if c.startswith("file_")]:
            fname = subject[col]
            if pd.isna(fname):
                continue
            raw = _read_session(os.path.join(study_dir, fname))
            frames.append(
                pd.DataFrame(
                    {
                        "subject_id": subject["subject_id"],
                        "study": subject["study"],
                        "session_id": fname.removesuffix(".csv"),
                        "key": raw["key"],
                        "press_ts": raw["press"],
                        "release_ts": raw["release"],
                        "label": subject["label"],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def load_tappy(raw_dir):
    """Not written yet. Tappy has a different layout, so it waits until the data is here."""
    raise NotImplementedError


def clean(df):
    """Drop invalid keystrokes. Return (clean_df, n_dropped, pct_dropped).

    Same rules as the dataset authors' sanityCheck():
      - press or release time not positive (the first row of a file is often
        a broken timestamp, e.g. a 1.4-billion-second hold time)
      - hold time outside [0, 5) seconds
      - press time earlier than one already seen in the same session
        (a key released after later keys, mostly Shift held while typing)
    """
    hold = df["release_ts"] - df["press_ts"]
    bad = (df["press_ts"] <= 0) | (df["release_ts"] <= 0) | (hold < 0) | (hold >= 5)

    by_session = df.groupby("session_id", sort=False)["press_ts"]
    prior_max = by_session.cummax().groupby(df["session_id"], sort=False).shift(1)
    bad |= df["press_ts"] < prior_max

    n_dropped = int(bad.sum())
    return df[~bad].reset_index(drop=True), n_dropped, 100 * n_dropped / len(df)


def typing_keys_only(df):
    """Drop mouse events, Shift/Alt/Control and BackSpace.

    Modifiers are held for a long time on purpose, so left in they swamp the
    hold-time distribution. BackSpace is an error correction, not normal typing.
    The authors' helper (getDataFiltHelper) removes the same three groups.
    """
    key = df["key"]
    drop = key.str.startswith("mouse") | key.str.match(r"^(Shift|Alt|Control)") | (key == "BackSpace")
    return df[~drop].reset_index(drop=True)
