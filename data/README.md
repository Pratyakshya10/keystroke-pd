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

### Tappy

Download `Archived-Data.zip` and `Archived-users.zip` from PhysioNet and extract
both into `data/raw/tappy/` so it holds the `Tappy Data/` folder (622 monthly
files) and the `User_*.txt` label files. Open access, Open Data Commons
Attribution License.

What the files contain, as loaded by `load_tappy()`:

- Data rows are tab separated with no header: user, date (YYMMDD), time of the key
  press, hand (`L`, `R`, or `S` for space), hold time in **milliseconds**, direction,
  latency, flight time. There are **no key names**, only the hand.
- Labels come from `Parkinsons: True/False` in each user file. They are
  self-reported, and the PD group is established and mostly treated (112 of 169
  take levodopa), unlike a clinic study of early disease.
- 227 users have a label file; 169 report PD and 58 do not. 49 users have data but
  no label file and are left out, and 10 labelled users have no data. That leaves
  217 users (162 PD, 55 control) and 9,014,543 keystrokes.
- 328 rows could not be parsed and were dropped. The same sanity rules as for
  neuroQWERTY then remove another 2.07% of keystrokes (about 13,400 holds are zero
  or negative, down to roughly minus 24 hours, which looks like a midnight
  rollover in the recorder; 642 are over 5 s).
- Data per user varies enormously: median 4,611 keystrokes, but 47 users have
  fewer than 500 and one has over a million. neuroQWERTY subjects have a median of
  1,702 and at least 299 after cleaning.
- The recorder clock is coarse. The most common hold values (62.5, 70.3, 78.1,
  85.9, 93.8, 101.6, 109.4, 125.0 ms) are multiples of 1/128 s = 7.8125 ms. About
  62% of all holds sit on that grid, and the share differs between users
  (most between 50% and 90%), so the resolution is not the same for everyone.
- One user has a single distinct hold value and a few others have only a handful,
  almost all because they have only a few keystrokes.
