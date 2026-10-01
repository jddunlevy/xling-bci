"""Synthetic recordings with the documented shape of each real dataset.

Lets the pipeline run and be tested before either dataset is downloaded. These
are not a model of EEG and the accuracies they produce mean nothing on their own.

Each class gets a spatial pattern, each subject its own mixing of those patterns,
and each language group a shared bias on top. So within-subject decoding should
come out high, cross-subject lower, and cross-language lower still. The tests
check that ordering.
"""

from __future__ import annotations

import mne
import numpy as np

from .datasets import Recording
from .labels import DIRECTIONAL_CLASSES, NIETO_CLASSES

mne.set_log_level("ERROR")


def _montage_channels(montage_name: str, n: int) -> tuple[list[str], mne.channels.DigMontage]:
    montage = mne.channels.make_standard_montage(montage_name)
    names = [ch for ch in montage.ch_names if ch not in ("Iz",)][:n]
    return names, montage


def _make_epochs(
    data: np.ndarray, ch_names: list[str], montage, sfreq: float, tmin: float, labels
) -> mne.Epochs:
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    classes = sorted(set(labels))
    event_id = {c: i + 1 for i, c in enumerate(classes)}
    events = np.column_stack(
        [
            np.arange(len(labels)) * int(sfreq),
            np.zeros(len(labels), dtype=int),
            np.asarray([event_id[c] for c in labels], dtype=int),
        ]
    )
    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=tmin)
    epochs.set_montage(montage, on_missing="ignore")
    return epochs


def _subject_data(
    rng: np.random.Generator,
    classes: tuple[str, ...],
    n_per_class: int,
    n_channels: int,
    n_times: int,
    group_bias: np.ndarray,
    separability: float,
    trial_jitter: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    n_sources = len(classes)

    # Class-specific temporal source, shared across subjects: the signal a decoder
    # could in principle transfer.
    t = np.linspace(0, 1, n_times)
    sources = np.stack([np.sin(2 * np.pi * (3 + 2 * k) * t) for k in range(n_sources)])

    # Subject-specific projection from sources to sensors, plus a group-level bias
    # so that same-language subjects are more alike than cross-language ones.
    mixing = rng.normal(0, 1, size=(n_channels, n_sources)) + group_bias

    X, y = [], []
    for k, label in enumerate(classes):
        for _ in range(n_per_class):
            # Trial-to-trial perturbation of the spatial pattern. Without it the
            # within-subject problem is linearly separable and pins at 1.0, which
            # would make the smoke test blind to regressions.
            pattern = mixing[:, k] + rng.normal(0, trial_jitter, size=n_channels)
            amplitude = separability * (0.8 + 0.4 * rng.random())
            signal = np.outer(pattern, sources[k]) * amplitude
            noise = rng.normal(0, 1.0, size=(n_channels, n_times))
            noise = np.cumsum(noise, axis=1) / np.sqrt(n_times)  # pink-ish drift
            X.append((signal + noise) * 1e-6)  # volts, so MNE scaling is sane
            y.append(label)

    return np.asarray(X), np.asarray(y)


def directional_like(
    n_russian: int = 12,
    n_spanish: int = 10,
    n_per_class: int = 24,
    separability: float = 0.22,
    seed: int = 17,
) -> list[Recording]:
    """Stand-in for the 2026 directional-word dataset: 38 ch, 500 Hz, -0.5 to 1.0 s."""
    rng = np.random.default_rng(seed)
    ch_names, montage = _montage_channels("standard_1020", 38)
    n_times = 750

    out: list[Recording] = []
    for language, count in (("russian", n_russian), ("spanish", n_spanish)):
        bias = rng.normal(0, 0.6, size=(38, len(DIRECTIONAL_CLASSES)))
        for i in range(count):
            X, y = _subject_data(
                rng,
                DIRECTIONAL_CLASSES,
                n_per_class,
                38,
                n_times,
                bias,
                separability,
            )
            epochs = _make_epochs(X, ch_names, montage, 500.0, -0.5, y)
            out.append(
                Recording(
                    subject=f"{language[:3]}-sub{i}",
                    dataset="directional",
                    language=language,
                    epochs=epochs,
                    labels=y,
                    meta={"synthetic": True},
                )
            )
    return out


def nieto_like(
    n_subjects: int = 10,
    n_per_class: int = 30,
    separability: float = 0.22,
    seed: int = 23,
) -> list[Recording]:
    """Stand-in for ds003626: BioSemi 128 naming, 1024 Hz, four classes."""
    rng = np.random.default_rng(seed)
    ch_names, montage = _montage_channels("biosemi128", 128)
    n_times = 1024
    bias = rng.normal(0, 0.6, size=(128, len(NIETO_CLASSES)))

    out: list[Recording] = []
    for i in range(n_subjects):
        X, y = _subject_data(
            rng, NIETO_CLASSES, n_per_class, 128, n_times, bias, separability
        )
        epochs = _make_epochs(X, ch_names, montage, 1024.0, -0.5, y)
        out.append(
            Recording(
                subject=f"nieto-sub{i}",
                dataset="nieto",
                language="spanish",
                epochs=epochs,
                labels=y,
                meta={"synthetic": True},
            )
        )
    return out
