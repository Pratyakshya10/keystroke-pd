# Data

Raw data is not committed to this repo.

## neuroQWERTY MIT-CSXPD
PhysioNet — search "neuroQWERTY MIT-CSXPD". Download to `data/raw/neuroqwerty/`.
The dataset ships its own reference loader (`nqDataLoader.py`). Read it before
writing your own parser.

## Tappy
PhysioNet: https://physionet.org/content/tappy/1.0.0/
Download to `data/raw/tappy/`.

## Cleaning notes

neuroQWERTY as loaded by `src/load.py`: 85 subjects (42 PD, 43 control), 116
sessions, 177,513 keystrokes. Subject IDs do not collide between the two studies
and every file named in the label files exists.

**Sanity rules** (same as the authors' `sanityCheck`): 3,570 keystrokes dropped,
**2.01%**. Rates per rule, each as a share of all keystrokes and overlapping:

| Rule | Share |
|---|---|
| press or release time not positive | 0.46% |
| hold time outside [0, 5) s | 0.66% |
| press time earlier than one already seen in the session | 1.99% |

The drop rate is the same in both groups (session average 1.79% control, 1.73%
PD), so cleaning does not favour one class. No session lost more than 6.1%.

**Non-typing keys** removed afterwards: mouse events, Shift/Alt/Control and
BackSpace, another 6,018 keystrokes (3.46% of what was left). Overall 5.40% of the
raw keystrokes are removed, leaving 167,925.

Things worth knowing:

- The first row of a session file can hold a broken timestamp (a hold time of
  about 1.4 billion seconds). The positive-time rule catches it.
- Left in, Shift holds alone create the long tail of the hold-time distribution.
  In one sample session, removing 5 modifier rows (0.5%) cut the hold-time std
  from 0.063 s to 0.024 s.
- The non-monotonic rule also drops some real letters: a key released after a
  later key was pressed (finger overlap) looks like press times going backwards.
  It is kept to match the authors, but it is a limitation.
- After cleaning, 67 keystrokes (0.04%) are still held longer than 0.5 s, mostly
  space, arrow keys and Return. 58 of the 67 come from PD subjects. They stay in,
  so skew and kurtosis are sensitive to them.
- The text is partly Spanish (`ntilde`, `exclamdown` appear among the keys).
- `updrs108` separates the groups perfectly (controls 0 to 6, PD 7 to 39.5), so it
  defines the label and is not used as an independent comparison.
- `afTap` is missing for 9 of the 31 CS1PD subjects.
