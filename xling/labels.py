"""Class vocabulary shared across datasets, and the per-dataset event maps.

The comparison is between concepts, not words, so event codes are converted to
concept names at load time and nothing downstream sees the original spelling.
"""

# The six directional concepts the 2026 directional-word dataset recorded in both
# Russian and Spanish. "next" exists for Russian speakers only and is therefore
# excluded from every cross-language analysis.
DIRECTIONAL_CLASSES = ("up", "down", "left", "right", "forward", "back")

# Nieto et al. recorded four of the six. Cross-dataset work is limited to these.
NIETO_CLASSES = ("up", "down", "left", "right")

# Concepts available when both datasets appear in the same analysis.
SHARED_CLASSES = tuple(c for c in DIRECTIONAL_CLASSES if c in NIETO_CLASSES)

RUSSIAN_ONLY = ("next",)

# Event-name suffix convention in the directional dataset: 1 = overt, 2 = covert.
# Overt trials are useful as a sanity check: they decode far better than covert
# ones, so an overt run near chance means something is wrong with the pipeline.
CONDITIONS = {"overt": "1", "covert": "2"}

# Non-trial markers in the directional dataset.
NON_TRIAL_MARKERS = ("GO", "GZ")


def parse_directional_event(name: str) -> tuple[str, str] | None:
    """Turn a directional-dataset event name into (concept, condition).

    Returns None for block-onset and rest markers, and for the Russian-only
    "next" class. Event IDs differ between the two language groups, so the name
    is the only reliable key.
    """
    upper = name.strip().upper()
    if upper in NON_TRIAL_MARKERS:
        return None
    if not upper or upper[-1] not in ("1", "2"):
        return None

    stem, suffix = upper[:-1], upper[-1]
    concept = stem.lower()
    if concept in RUSSIAN_ONLY:
        return None
    if concept not in DIRECTIONAL_CLASSES:
        return None

    condition = "overt" if suffix == "1" else "covert"
    return concept, condition
