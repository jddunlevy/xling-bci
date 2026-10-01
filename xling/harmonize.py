"""Put recordings from different rigs into one comparable space.

The two datasets use different amplifiers, montages and sampling rates, so any
accuracy difference between them is confounded with all three until this step
removes what it can. What it cannot remove is listed in the README.

Order matters: filter, then crop, then resample, then select channels. Filtering
after cropping leaks edge artifacts into the window of interest.
"""

from __future__ import annotations

import numpy as np

from .datasets import Recording


def to_array(
    recordings: list[Recording],
    sfreq: float,
    window: tuple[float, float],
    bandpass: tuple[float, float] | None,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack recordings into (X, y, subject) with one common space.

    X is (n_trials, n_channels, n_times), y holds concept names, subject holds the
    recording each trial came from so grouped cross-validation can hold a whole
    subject out.
    """
    xs, ys, groups = [], [], []
    for rec in recordings:
        epochs = rec.epochs.copy()
        if bandpass is not None:
            epochs.filter(bandpass[0], bandpass[1], picks="eeg", verbose="ERROR")
        epochs.crop(*window)
        if epochs.info["sfreq"] != sfreq:
            epochs.resample(sfreq, verbose="ERROR")
        if channels is not None:
            # pick() does not guarantee order, and covariance matrices are only
            # comparable across recordings if channel order is identical.
            epochs.pick(channels)
            epochs.reorder_channels(list(channels))

        data = epochs.get_data(copy=True)
        xs.append(data)
        ys.append(rec.labels)
        groups.append(np.full(len(rec.labels), rec.subject))

    n_times = min(x.shape[-1] for x in xs)
    n_chans = {x.shape[1] for x in xs}
    if len(n_chans) != 1:
        raise ValueError(
            f"channel counts differ after harmonizing: {sorted(n_chans)}. "
            "Pass an explicit `channels` list from shared_channels()."
        )

    X = np.concatenate([x[..., :n_times] for x in xs], axis=0)
    return X, np.concatenate(ys), np.concatenate(groups)


def shared_channels(recordings: list[Recording]) -> list[str]:
    """Channel names present in every recording, as written.

    The directional dataset uses 10-10 names. Nieto's BioSemi 128 cap labels
    electrodes A1-D32, so the literal intersection of the two is empty and this
    raises rather than silently producing a one-channel analysis. Use
    `match_by_position` for the cross-dataset arm.
    """
    sets = [set(rec.epochs.copy().pick("eeg").ch_names) for rec in recordings]
    common = set.intersection(*sets) if sets else set()

    if not common:
        raise ValueError(
            "no channel names shared across recordings. If this is a cross-dataset "
            "comparison, the two montages use different naming conventions "
            "(10-10 vs BioSemi A1-D32) and you need match_by_position() instead."
        )

    # Preserve the first recording's order so channel order is deterministic.
    first = recordings[0].epochs.copy().pick("eeg").ch_names
    return [ch for ch in first if ch in common]


def match_by_position(
    target: Recording, source: Recording, max_distance: float = 0.025
) -> dict[str, str]:
    """Map target channels to the nearest source channel by montage position.

    Returns {target_name: source_name}. Channels with no source electrode within
    `max_distance` metres are dropped, so the result is usually smaller than the
    target montage. This is a spatial approximation, not an equivalence, and any
    cross-dataset result that depends on it has to say so.
    """
    t_names, t_pos = _positions(target)
    s_names, s_pos = _positions(source)

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for name, pos in zip(t_names, t_pos):
        distances = np.linalg.norm(s_pos - pos, axis=1)
        order = np.argsort(distances)
        for idx in order:
            if distances[idx] > max_distance:
                break
            candidate = s_names[idx]
            if candidate in used:
                continue
            mapping[name] = candidate
            used.add(candidate)
            break

    if not mapping:
        raise ValueError(
            "no channels matched by position. Check that both recordings have a "
            "montage set (epochs.get_montage() must not be None)."
        )
    return mapping


def _positions(rec: Recording) -> tuple[list[str], np.ndarray]:
    epochs = rec.epochs.copy().pick("eeg")
    montage = epochs.get_montage()
    if montage is None:
        raise ValueError(f"{rec.subject} has no montage; cannot match by position")

    coords = montage.get_positions()["ch_pos"]
    names = [ch for ch in epochs.ch_names if ch in coords and np.isfinite(coords[ch]).all()]
    if not names:
        raise ValueError(f"{rec.subject} montage has no finite channel positions")
    return names, np.asarray([coords[ch] for ch in names])
