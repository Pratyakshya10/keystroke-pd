"""Parse raw keystroke logs into a common schema.

Target schema (one row per keystroke):
    subject_id : str
    session_id : str
    key        : str
    press_ts   : float   seconds
    release_ts : float   seconds
    label      : int     1 = PD, 0 = control

WRITE THESE YOURSELF. Do not paste in an implementation.
Open the raw files and look at them before writing any parsing code.
"""


def load_neuroqwerty(raw_dir):
    """Return a DataFrame in the target schema."""
    raise NotImplementedError


def load_tappy(raw_dir):
    """Return a DataFrame in the target schema."""
    raise NotImplementedError


def clean(df):
    """Drop invalid rows. Return (clean_df, n_dropped, pct_dropped)."""
    raise NotImplementedError
