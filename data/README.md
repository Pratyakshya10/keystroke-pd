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
Record what you dropped and why. The reference loader removes rows with
non-positive or non-monotonic timestamps and hold times outside [0, 5) seconds.
Log your own drop percentage here — it goes in the README.
