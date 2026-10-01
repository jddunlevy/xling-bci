"""End-to-end check on synthetic data. Runs without either dataset downloaded.

What this verifies is the *ordering* the pipeline must reproduce, not accuracy
values. Synthetic within-subject decoding sits near ceiling by construction,
because each synthetic class carries its own frequency and covariance separates
that trivially. The generator exists to exercise the machinery.

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from xling import decode, harmonize, report, synthetic


def build(n_per_class: int = 10):
    recs = synthetic.directional_like(n_russian=4, n_spanish=4, n_per_class=n_per_class)
    channels = harmonize.shared_channels(recs)
    X, y, groups = harmonize.to_array(
        recs,
        sfreq=config.TARGET_SFREQ,
        window=config.TARGET_WINDOW,
        bandpass=config.BANDPASS,
        channels=channels,
    )
    language = np.asarray([r.language for r in recs for _ in range(r.n_trials)])
    return X, y, groups, language


def test_shapes_and_alignment():
    X, y, groups, _ = build()
    assert X.ndim == 3 and X.shape[1] == 38
    assert len(y) == len(groups) == X.shape[0]

    F = decode.features(X, groups, align=True)
    n_ch = X.shape[1]
    assert F.shape == (X.shape[0], n_ch * (n_ch + 1) // 2)
    assert np.isfinite(F).all()
    print(f"ok  shapes          X={X.shape} F={F.shape}")


def test_recentering_is_per_subject():
    """Each subject's covariances must come out centered near identity.

    The relevant mean is the Riemannian one. Whitening by the geometric mean
    leaves the arithmetic mean far from identity, so checking that instead would
    fail on correct code.
    """
    from pyriemann.utils.mean import mean_covariance

    X, _, groups, _ = build()
    C = decode.recenter_by_group(decode.covariances(X), groups)
    for g in np.unique(groups):
        mean = mean_covariance(C[groups == g], metric="riemann")
        off = np.abs(mean - np.eye(mean.shape[0]))
        assert off.max() < 1e-3, f"{g} not centered, max deviation {off.max():.3e}"
    print("ok  recentering     every subject centered near identity")


def test_transfer_ordering():
    """within subject > within language > across language, and across ~ chance."""
    X, y, groups, language = build()
    F = decode.features(X, groups, align=True)
    chance = decode.chance_level(y)

    within_subj = np.mean(list(decode.within_subject(F, y, groups).values()))

    within_lang, across_lang = {}, {}
    for lang in sorted(set(language)):
        same = np.unique(groups[language == lang])
        other = np.unique(groups[language != lang])
        within_lang.update(
            decode.leave_one_subject_out(F, y, groups, train_groups=same, test_groups=same)
        )
        across_lang.update(
            decode.leave_one_subject_out(F, y, groups, train_groups=other, test_groups=same)
        )

    w = np.mean(list(within_lang.values()))
    a = np.mean(list(across_lang.values()))

    assert within_subj > w, f"within-subject {within_subj:.3f} <= within-language {w:.3f}"
    assert w > a, f"within-language {w:.3f} <= across-language {a:.3f}"
    assert within_subj > chance * 2, "pipeline is not decoding at all"

    contrast = report.language_contrast(within_lang, across_lang, chance)
    assert set(contrast) >= {"between_language_gap", "within_language_sd"}
    assert report.interpret(contrast)
    print(
        f"ok  ordering        subject={within_subj:.3f} > "
        f"language={w:.3f} > across={a:.3f} (chance={chance:.3f})"
    )


def test_cross_dataset_channel_matching():
    """10-10 names and BioSemi A1-D32 names must match by position, not string."""
    directional = synthetic.directional_like(n_russian=1, n_spanish=1, n_per_class=4)
    nieto = synthetic.nieto_like(n_subjects=1, n_per_class=4)

    try:
        harmonize.shared_channels([directional[0], nieto[0]])
        raise AssertionError("expected name-based matching to fail across montages")
    except ValueError:
        pass

    mapping = harmonize.match_by_position(directional[0], nieto[0])
    assert len(mapping) >= 10, f"only {len(mapping)} channels matched"
    assert len(set(mapping.values())) == len(mapping), "a source channel was reused"
    print(f"ok  channel match   {len(mapping)} channels matched by position")


if __name__ == "__main__":
    tests = [
        test_shapes_and_alignment,
        test_recentering_is_per_subject,
        test_transfer_ordering,
        test_cross_dataset_channel_matching,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} passed")
