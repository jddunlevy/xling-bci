"""Paths and analysis constants. Edit DATA_ROOT after downloading the datasets."""

from pathlib import Path

REPO = Path(__file__).parent
DATA_ROOT = REPO / "data"

# Zenodo release accompanying Sci Data 10.1038/s41597-026-07809-9.
# Expected layout: raw/{russian,spanish}/sub*/, preprocessed/{russian,spanish}/, metadata/
DIRECTIONAL_ROOT = DATA_ROOT / "directional"

# OpenNeuro ds003626, Nieto et al. 2022. Use the derivatives/ preprocessed epochs.
NIETO_ROOT = DATA_ROOT / "ds003626"

RESULTS = REPO / "results"

# Common space every dataset is resampled into before any cross-dataset comparison.
TARGET_SFREQ = 250.0
TARGET_WINDOW = (-0.4, 1.0)  # seconds relative to cue, inside both datasets' epochs
BANDPASS = (1.0, 40.0)

RANDOM_STATE = 17
