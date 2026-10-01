"""Dataset loaders.

Each loader returns `Recording` objects, one per subject per session, reduced to
the condition and concepts asked for. Loaders do no filtering, resampling or
channel selection; that happens in harmonize.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

from .labels import parse_directional_event

mne.set_log_level("ERROR")


@dataclass
class Recording:
    """One subject's epochs, with the metadata the analyses group by."""

    subject: str
    dataset: str
    language: str
    epochs: mne.Epochs
    labels: np.ndarray  # concept names, one per epoch
    meta: dict = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        classes = ", ".join(sorted(set(self.labels)))
        return (
            f"Recording({self.dataset}/{self.subject} {self.language} "
            f"n={self.n_trials} [{classes}])"
        )


# --------------------------------------------------------------------------
# Directional-word dataset (Zenodo, Sci Data 10.1038/s41597-026-07809-9)
# 22 subjects: 12 native Russian (sub1-sub12), 10 native Spanish (sub0-sub9).
# preprocessed/<language>/ holds epochs in .fif with event files in .xlsx.
# 38 channels, 500 Hz, epochs 1.5 s spanning -0.5 to +1.0 around the marker.
# --------------------------------------------------------------------------

DIRECTIONAL_LANGUAGES = ("russian", "spanish")


def load_directional(
    root: Path,
    languages: tuple[str, ...] = DIRECTIONAL_LANGUAGES,
    condition: str = "covert",
    concepts: tuple[str, ...] | None = None,
) -> list[Recording]:
    root = Path(root)
    metadata = _read_subject_metadata(root)
    out: list[Recording] = []

    for language in languages:
        folder = root / "preprocessed" / language
        if not folder.is_dir():
            raise FileNotFoundError(
                f"expected {folder}. Download the Zenodo release for Sci Data "
                f"10.1038/s41597-026-07809-9 and point config.DIRECTIONAL_ROOT at it."
            )

        for fif in sorted(folder.glob("**/*epo.fif")) or sorted(folder.glob("**/*.fif")):
            subject = _subject_from_path(fif)
            epochs = mne.read_epochs(fif, preload=True)
            keep, names = _select_directional(epochs, condition, concepts)
            if not len(keep):
                continue
            out.append(
                Recording(
                    subject=f"{language[:3]}-{subject}",
                    dataset="directional",
                    language=language,
                    epochs=epochs[keep],
                    labels=names,
                    meta=metadata.get(subject, {}),
                )
            )

    if not out:
        raise RuntimeError(f"no usable recordings under {root}")
    return out


def _select_directional(
    epochs: mne.Epochs, condition: str, concepts: tuple[str, ...] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Indices and concept names for the trials matching condition and concepts.

    Event *IDs* differ between the Russian and Spanish groups, so selection goes
    through event names only. That is also why this cannot be an event_id dict.
    """
    inv = {code: name for name, code in epochs.event_id.items()}
    codes = epochs.events[:, 2]

    keep, names = [], []
    for i, code in enumerate(codes):
        parsed = parse_directional_event(inv.get(code, ""))
        if parsed is None:
            continue
        concept, cond = parsed
        if cond != condition:
            continue
        if concepts is not None and concept not in concepts:
            continue
        keep.append(i)
        names.append(concept)

    return np.asarray(keep, dtype=int), np.asarray(names)


def _read_subject_metadata(root: Path) -> dict:
    path = root / "metadata" / "subject_metadata.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        return raw
    # Some releases ship a list of records keyed by an id field.
    return {str(r.get("subject", r.get("id", i))): r for i, r in enumerate(raw)}


def _subject_from_path(path: Path) -> str:
    for part in (path.stem, *reversed(path.parts)):
        lowered = str(part).lower()
        if lowered.startswith("sub"):
            return lowered.split("_")[0].split("-")[-1] if "-" in lowered else lowered
    return path.stem


# --------------------------------------------------------------------------
# Nieto et al. 2022, OpenNeuro ds003626.
# 10 native Spanish speakers, 3 sessions each, 128-channel BioSemi at 1024 Hz.
# Inner / pronounced / visualized conditions; classes arriba, abajo, derecha,
# izquierda. Used here as the same-language, different-hardware control.
# --------------------------------------------------------------------------

NIETO_CONCEPT_BY_CODE = {31: "up", 32: "down", 33: "right", 34: "left"}
NIETO_SPANISH_WORDS = {"arriba": "up", "abajo": "down", "derecha": "right", "izquierda": "left"}
NIETO_INNER_SPEECH = 1  # condition code for the inner-speech paradigm


def load_nieto(root: Path, concepts: tuple[str, ...] | None = None) -> list[Recording]:
    """Load the inner-speech epochs from the dataset's derivatives/.

    Nieto ships preprocessed epochs plus an events array per subject-session. The
    events array encodes condition and class in separate columns, and the column
    order has changed between releases, so the mapping is resolved defensively and
    a clear error is raised rather than guessed at.
    """
    root = Path(root)
    derivatives = root / "derivatives"
    if not derivatives.is_dir():
        raise FileNotFoundError(
            f"expected {derivatives}. Fetch ds003626 derivatives with the helper "
            f"scripts at https://github.com/N-Nieto/Inner_Speech_Dataset"
        )

    out: list[Recording] = []
    for fif in sorted(derivatives.glob("**/*eeg-epo.fif")) or sorted(derivatives.glob("**/*epo.fif")):
        subject = _nieto_subject(fif)
        events_path = _nieto_events_path(fif)
        epochs = mne.read_epochs(fif, preload=True)
        events = np.load(events_path) if events_path else None
        if events is None:
            raise FileNotFoundError(
                f"no events array beside {fif}; expected a *events.dat or *events.npy"
            )

        keep, names = _select_nieto(events, concepts)
        if not len(keep):
            continue
        out.append(
            Recording(
                subject=f"nieto-{subject}",
                dataset="nieto",
                language="spanish",
                epochs=epochs[keep],
                labels=names,
                meta={"session": _nieto_session(fif)},
            )
        )

    if not out:
        raise RuntimeError(f"no usable recordings under {derivatives}")
    return out


def _select_nieto(events: np.ndarray, concepts: tuple[str, ...] | None):
    """Pick inner-speech trials and map class codes to concept names.

    The events array is (n_trials, 4): time, class code, condition code, session.
    Only the class and condition columns are used, and both are validated against
    the documented code sets before anything is selected.
    """
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError(f"unexpected Nieto events shape {events.shape}")

    class_col, cond_col = _resolve_nieto_columns(events)

    keep, names = [], []
    for i, row in enumerate(events):
        if int(row[cond_col]) != NIETO_INNER_SPEECH:
            continue
        concept = NIETO_CONCEPT_BY_CODE.get(int(row[class_col]))
        if concept is None:
            continue
        if concepts is not None and concept not in concepts:
            continue
        keep.append(i)
        names.append(concept)

    return np.asarray(keep, dtype=int), np.asarray(names)


def _resolve_nieto_columns(events: np.ndarray) -> tuple[int, int]:
    """Find which column holds class codes and which holds condition codes."""
    class_col = cond_col = None
    for col in range(events.shape[1]):
        values = set(np.unique(events[:, col]).astype(int).tolist())
        if values and values <= set(NIETO_CONCEPT_BY_CODE):
            class_col = col
        elif values and values <= {1, 2, 3}:
            cond_col = col

    if class_col is None or cond_col is None:
        raise ValueError(
            "could not identify class/condition columns in the Nieto events array. "
            f"Columns held: {[sorted(set(np.unique(events[:, c]).astype(int).tolist()))[:6] for c in range(events.shape[1])]}. "
            "Check the release's documentation and set the columns explicitly."
        )
    return class_col, cond_col


def _nieto_subject(path: Path) -> str:
    for part in path.parts:
        if part.lower().startswith("sub-"):
            return part.split("-", 1)[1]
    return path.stem


def _nieto_session(path: Path) -> str:
    for part in path.parts:
        if part.lower().startswith("ses-"):
            return part.split("-", 1)[1]
    return "1"


def _nieto_events_path(fif: Path) -> Path | None:
    stem = fif.name.replace("eeg-epo.fif", "").replace("epo.fif", "")
    for pattern in (f"{stem}events.dat", f"{stem}events.npy", "*events.dat", "*events.npy"):
        hits = sorted(fif.parent.glob(pattern))
        if hits:
            return hits[0]
    return None
