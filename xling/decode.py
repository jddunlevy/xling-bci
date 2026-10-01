"""Decoding pipeline and the four evaluation regimes.

Covariance matrices, tangent space, logistic regression. The standard simple
baseline rather than a deep network: hard to beat on samples this size, nothing
to tune, and fast enough that a weak transfer number is more likely to be about
transfer than about how long it trained.

Per-subject re-centering matters more than the classifier does. Cross-subject
EEG transfer without it sits near chance for reasons unrelated to language, so
cross-subject numbers are worth reporting both ways.

Re-centering also buys speed. Once each subject's covariances are whitened to
their own Riemannian mean, the identity is the correct tangent-space reference,
so the projection is computed once for the whole dataset instead of an iterative
mean refitted inside every fold.
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


def covariances(X: np.ndarray, estimator: str = "oas") -> np.ndarray:
    """Trial-wise spatial covariance. OAS shrinkage, because trials are short."""
    return Covariances(estimator=estimator).transform(X)


def recenter(C: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Whiten covariances by a reference mean so the set is centered at identity.

    Uses no labels, so applying it to held-out data is not leakage.
    """
    M = mean_covariance(C, metric="riemann") if reference is None else reference
    W = invsqrtm(M)
    return W @ C @ W


def recenter_by_group(C: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Re-center each subject independently. The domain-adaptation step."""
    out = np.empty_like(C)
    for g in np.unique(groups):
        mask = groups == g
        out[mask] = recenter(C[mask])
    return out


def features(
    X: np.ndarray, groups: np.ndarray | None = None, align: bool = True
) -> np.ndarray:
    """Epochs to tangent-space vectors, the representation every regime shares.

    With `align`, each subject is whitened to its own mean and the reference is
    the identity. Without it, the reference is the Riemannian mean of everything,
    which is the no-domain-adaptation baseline.
    """
    C = covariances(X)
    if align:
        if groups is None:
            raise ValueError("align=True needs `groups` to re-center per subject")
        C = recenter_by_group(C, groups)
        reference = np.eye(C.shape[-1])
    else:
        reference = mean_covariance(C, metric="riemann")
    return tangent_space(C, reference)


def make_classifier(seed: int = 17) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)


def chance_level(y: np.ndarray) -> float:
    return 1.0 / len(np.unique(y))


# --------------------------------------------------------------------------
# Regime 1: within subject. The ceiling, and the proof the pipeline works.
# --------------------------------------------------------------------------


def within_subject(
    F: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 17
) -> dict[str, float]:
    scores: dict[str, float] = {}

    for g in np.unique(groups):
        mask = groups == g
        Fg, yg = F[mask], y[mask]
        if min(np.bincount(_codes(yg))) < n_splits:
            continue

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for train, test in cv.split(Fg, yg):
            clf = make_classifier(seed)
            clf.fit(Fg[train], yg[train])
            folds.append(balanced_accuracy_score(yg[test], clf.predict(Fg[test])))
        scores[str(g)] = float(np.mean(folds))

    return scores


# --------------------------------------------------------------------------
# Regimes 2 and 3: leave one subject out, inside a language or across languages.
# Same function both times; only the training pool changes.
# --------------------------------------------------------------------------


def leave_one_subject_out(
    F: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_groups: np.ndarray | None = None,
    test_groups: np.ndarray | None = None,
    seed: int = 17,
) -> dict[str, float]:
    """Score each held-out subject, training on the rest.

    `train_groups` restricts the training pool, which is how the across-language
    arm is built: hold out a Spanish speaker, train only on Russian speakers.
    `test_groups` restricts which subjects are scored, so the across-language arm
    does not waste fits on subjects it will discard.
    """
    pool = np.unique(groups) if train_groups is None else np.unique(train_groups)
    targets = np.unique(groups) if test_groups is None else np.unique(test_groups)
    scores: dict[str, float] = {}

    for g in targets:
        train_mask = np.isin(groups, pool) & (groups != g)
        test_mask = groups == g
        if not train_mask.any() or not test_mask.any():
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue

        clf = make_classifier(seed)
        clf.fit(F[train_mask], y[train_mask])
        scores[str(g)] = float(
            balanced_accuracy_score(y[test_mask], clf.predict(F[test_mask]))
        )

    return scores


# --------------------------------------------------------------------------
# Regime 4: train on one dataset, test on the other. The hardware control.
# --------------------------------------------------------------------------


def cross_dataset(
    F_train: np.ndarray,
    y_train: np.ndarray,
    F_test: np.ndarray,
    y_test: np.ndarray,
    g_test: np.ndarray,
    seed: int = 17,
) -> dict[str, float]:
    shared = sorted(set(y_train) & set(y_test))
    if len(shared) < 2:
        raise ValueError(f"datasets share too few classes: {shared}")

    tr = np.isin(y_train, shared)
    clf = make_classifier(seed)
    clf.fit(F_train[tr], y_train[tr])

    scores: dict[str, float] = {}
    for g in np.unique(g_test):
        mask = (g_test == g) & np.isin(y_test, shared)
        if not mask.any():
            continue
        scores[str(g)] = float(
            balanced_accuracy_score(y_test[mask], clf.predict(F_test[mask]))
        )
    return scores


# --------------------------------------------------------------------------
# Null distribution. Six classes puts chance at .167, but small per-subject test
# sets make the empirical null wider than that, so report both.
# --------------------------------------------------------------------------


def shuffle_null(
    F: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_permutations: int = 20,
    seed: int = 17,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_permutations):
        shuffled = y.copy()
        for g in np.unique(groups):
            mask = groups == g
            shuffled[mask] = rng.permutation(shuffled[mask])
        out.extend(leave_one_subject_out(F, shuffled, groups, seed=seed).values())
    return np.asarray(out)


def _codes(y: np.ndarray) -> np.ndarray:
    _, codes = np.unique(y, return_inverse=True)
    return codes
