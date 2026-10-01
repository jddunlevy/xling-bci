"""Per-subject scores to summary numbers.

The number of interest is not accuracy on its own. It is the between-language
transfer gap set against the spread between individuals who share a language,
since that comparison is what says whether the language label groups anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def summarize(scores: dict[str, float]) -> dict[str, float]:
    values = np.asarray(list(scores.values()), dtype=float)
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def language_contrast(
    within: dict[str, float], across: dict[str, float], chance: float
) -> dict[str, float]:
    """Gap and spread, computed explicitly rather than left to the reader.

    `within` is leave-one-subject-out trained on the same language group as the
    held-out subject. `across` is the same held-out subjects, trained on the other
    language group. Both are keyed by subject, so the gap is paired.
    """
    shared = sorted(set(within) & set(across))
    if not shared:
        raise ValueError("no subjects appear in both regimes")

    w = np.asarray([within[s] for s in shared])
    a = np.asarray([across[s] for s in shared])

    gap = float((w - a).mean())
    spread = float(w.std(ddof=1)) if w.size > 1 else 0.0

    return {
        "n_subjects": len(shared),
        "chance": chance,
        "within_language_mean": float(w.mean()),
        "across_language_mean": float(a.mean()),
        "between_language_gap": gap,
        "within_language_sd": spread,
        "gap_over_spread": float(gap / spread) if spread > 0 else float("nan"),
    }


def interpret(contrast: dict[str, float]) -> str:
    """One line saying which way the comparison came out. Not a conclusion."""
    gap = contrast["between_language_gap"]
    spread = contrast["within_language_sd"]

    if spread > 0 and gap < spread:
        return (
            "Variation between individuals who share a language exceeds the gap "
            "between language groups. The language label is not what separates "
            "these subjects, and these datasets record little else about them."
        )
    if gap >= spread > 0:
        return (
            "The between-language gap exceeds variation within a language group. "
            "Language background conditions transfer here."
        )
    return "Too few subjects to compare gap against spread."


def table(rows: dict[str, dict[str, float]]) -> str:
    """Plain fixed-width table. No dependencies, reads fine in a terminal."""
    header = f"{'regime':<34}{'n':>4}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}"
    lines = [header, "-" * len(header)]
    for name, stats in rows.items():
        if not stats.get("n"):
            lines.append(f"{name:<34}{0:>4}{'':>9}{'':>8}{'':>8}{'':>8}")
            continue
        lines.append(
            f"{name:<34}{stats['n']:>4}{stats['mean']:>9.3f}"
            f"{stats['sd']:>8.3f}{stats['min']:>8.3f}{stats['max']:>8.3f}"
        )
    return "\n".join(lines)


def save(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
