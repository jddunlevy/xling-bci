"""Run the four decoding regimes and print the contrast the project argues from.

    python experiments/run.py --data synthetic
    python experiments/run.py --data real --regime 3

Regimes:
    1  within subject              ceiling, and proof the pipeline decodes anything
    2  within language, LOSO       transfer between subjects who share a language
    3  across language, LOSO       same held-out subjects, trained on the other group
    4  across dataset, same lang   Nieto Spanish vs directional Spanish; the rig control
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from xling import decode, harmonize, report, synthetic
from xling.datasets import load_directional, load_nieto
from xling.labels import DIRECTIONAL_CLASSES, SHARED_CLASSES


def load(source: str):
    if source == "synthetic":
        return synthetic.directional_like(), synthetic.nieto_like()
    directional = load_directional(
        config.DIRECTIONAL_ROOT, condition="covert", concepts=DIRECTIONAL_CLASSES
    )
    nieto = load_nieto(config.NIETO_ROOT, concepts=SHARED_CLASSES)
    return directional, nieto


def to_space(recordings, channels=None):
    return harmonize.to_array(
        recordings,
        sfreq=config.TARGET_SFREQ,
        window=config.TARGET_WINDOW,
        bandpass=config.BANDPASS,
        channels=channels,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--regime", choices=("all", "1", "2", "3", "4"), default="all")
    parser.add_argument("--no-align", action="store_true", help="skip re-centering")
    parser.add_argument("--null", action="store_true", help="run the label-shuffle null")
    args = parser.parse_args()
    align = not args.no_align
    seed = config.RANDOM_STATE

    directional, nieto = load(args.data)
    print(f"loaded {len(directional)} directional, {len(nieto)} nieto recordings")

    channels = harmonize.shared_channels(directional)
    X, y, groups = to_space(directional, channels)
    language = np.asarray(
        [rec.language for rec in directional for _ in range(rec.n_trials)]
    )
    print(f"X={X.shape} classes={sorted(set(y))} chance={decode.chance_level(y):.3f}")

    # One representation, shared by regimes 1 through 3. Alignment happens here
    # rather than inside each regime so no regime can silently differ on it.
    F = decode.features(X, groups, align=align)
    print(f"tangent features {F.shape} align={align}\n")

    want = {"1", "2", "3", "4"} if args.regime == "all" else {args.regime}
    rows: dict[str, dict] = {}
    payload: dict[str, object] = {"data": args.data, "align": align}

    if "1" in want:
        scores = decode.within_subject(F, y, groups, seed=seed)
        rows["1 within subject"] = report.summarize(scores)
        payload["within_subject"] = scores

    within_lang: dict[str, float] = {}
    across_lang: dict[str, float] = {}

    for lang in sorted(set(language)):
        same = np.unique(groups[language == lang])
        other = np.unique(groups[language != lang])

        if "2" in want:
            s = decode.leave_one_subject_out(
                F, y, groups, train_groups=same, test_groups=same, seed=seed
            )
            within_lang.update(s)
            rows[f"2 within language [{lang}]"] = report.summarize(s)

        if "3" in want:
            s = decode.leave_one_subject_out(
                F, y, groups, train_groups=other, test_groups=same, seed=seed
            )
            across_lang.update(s)
            rows[f"3 across language [-> {lang}]"] = report.summarize(s)

    if within_lang:
        payload["within_language"] = within_lang
    if across_lang:
        payload["across_language"] = across_lang

    if "4" in want:
        mapping = harmonize.match_by_position(directional[0], nieto[0])
        print(f"regime 4 matched {len(mapping)} channels by montage position")
        d_spanish = [r for r in directional if r.language == "spanish"]
        Xd, yd, gd = to_space(d_spanish, list(mapping.keys()))
        Xn, yn, gn = to_space(nieto, list(mapping.values()))
        n_times = min(Xd.shape[-1], Xn.shape[-1])

        Fn = decode.features(Xn[..., :n_times], gn, align=align)
        Fd = decode.features(Xd[..., :n_times], gd, align=align)
        scores = decode.cross_dataset(Fn, yn, Fd, yd, gd, seed=seed)
        rows["4 nieto -> directional (es)"] = report.summarize(scores)
        payload["cross_dataset"] = scores

    print(report.table(rows))

    if within_lang and across_lang:
        contrast = report.language_contrast(
            within_lang, across_lang, decode.chance_level(y)
        )
        payload["contrast"] = contrast
        print("\ncontrast")
        for key, value in contrast.items():
            formatted = f"{value:.3f}" if isinstance(value, float) else str(value)
            print(f"  {key:<24}{formatted}")
        print(f"\n{report.interpret(contrast)}")

    if args.null:
        null = decode.shuffle_null(F, y, groups, seed=seed)
        payload["null"] = {"mean": float(null.mean()), "p95": float(np.quantile(null, 0.95))}
        print(f"\nnull mean={null.mean():.3f} p95={np.quantile(null, 0.95):.3f}")

    out = report.save(payload, config.RESULTS / f"{args.data}-regime-{args.regime}.json")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
