"""Load the neuroQWERTY (MIT-CSXPD) and Tappy keystroke data into DataFrames.

load_tappy() is further down; Tappy has a different layout (see its docstring).

neuroQWERTY raw session files have no header and four columns: key, hold, release, press.
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
import glob
import os

import numpy as np
import pandas as pd

STUDIES = ["CS1PD", "CS2PD"]
RAW_COLUMNS = ["key", "hold", "release", "press"]
HANDS = ["L", "R", "S"]
DIRECTIONS = ["LL", "LR", "LS", "RL", "RR", "RS", "SL", "SR", "SS"]
HAND_TYPE = pd.CategoricalDtype(HANDS)
DIRECTION_TYPE = pd.CategoricalDtype(DIRECTIONS)


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


def load_tappy_users(raw_dir):
    """One row per user that has a User_*.txt file (key: value lines). label 1 = PD."""
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "User_*.txt"))):
        fields = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
        fields["subject_id"] = os.path.basename(path)[len("User_"):-len(".txt")]
        rows.append(fields)
    users = pd.DataFrame(rows)
    if not users["Parkinsons"].isin(["True", "False"]).all():
        raise ValueError("unexpected value in the Parkinsons field")
    users["label"] = (users["Parkinsons"] == "True").astype(int)
    return users


def load_tappy(raw_dir):
    """Tappy keystrokes for every user that has a label file, one row per keystroke.

    Data files have no header: user, date (YYMMDD), time (HH:MM:SS.mmm of the key
    press), hand (L/R/S for space), hold time in MILLISECONDS, direction, latency,
    flight. Unlike neuroQWERTY there are no key names, only the hand.

    Columns: subject_id, study, session_id (file name), hand, direction (hand of
    the previous key + hand of this one), press_ts and release_ts (seconds since
    2000-01-01, so the same feature code works on both datasets), latency_ms and
    flight_ms as recorded, and label. Rows whose date, time, hand, direction or
    hold time can't be parsed are dropped (a few lines have fields run together);
    their count is in df.attrs["n_malformed"].
    """
    labels = load_tappy_users(raw_dir).set_index("subject_id")["label"]
    epoch = pd.Timestamp("2000-01-01")
    frames, n_malformed = [], 0
    for path in sorted(glob.glob(os.path.join(raw_dir, "Tappy Data", "*.txt"))):
        session = os.path.basename(path)[: -len(".txt")]
        subject = session.split("_")[0]
        if subject not in labels.index:
            continue
        raw = pd.read_csv(path, sep="\t", header=None, usecols=[1, 2, 3, 4, 5, 6, 7], dtype=str,
                          names=["date", "time", "hand", "hold_ms", "direction", "latency_ms", "flight_ms"],
                          keep_default_na=False)
        hold_ms = pd.to_numeric(raw["hold_ms"], errors="coerce")
        day = pd.to_datetime(raw["date"], format="%y%m%d", errors="coerce")
        clock = raw["time"].str.extract(r"^(\d\d):(\d\d):(\d\d\.\d+)$").astype(float)
        ok = (hold_ms.notna() & day.notna() & clock.notna().all(axis=1)
              & raw["hand"].isin(HANDS) & raw["direction"].isin(DIRECTIONS))
        n_malformed += int((~ok).sum())
        raw, hold_ms, day, clock = raw[ok], hold_ms[ok], day[ok], clock[ok]
        press = (day - epoch).dt.days * 86400.0 + clock[0] * 3600 + clock[1] * 60 + clock[2]
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": subject,
                    "study": "Tappy",
                    "session_id": session,
                    "hand": raw["hand"].astype(HAND_TYPE),
                    "direction": raw["direction"].astype(DIRECTION_TYPE),
                    "press_ts": press,
                    "release_ts": press + hold_ms / 1000.0,
                    "latency_ms": pd.to_numeric(raw["latency_ms"], errors="coerce"),
                    "flight_ms": pd.to_numeric(raw["flight_ms"], errors="coerce"),
                    "label": int(labels[subject]),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df.attrs["n_malformed"] = n_malformed
    return df


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
