"""Leave-one-subject-out evaluation.

Why LOSO: with ~85 subjects, a random split puts the same person's keystrokes
in both train and test. The model then recognises the person, not the disease,
and AUC is inflated. LeaveOneGroupOut with subject_id as the group prevents it.

A single held-out subject has one label, so AUC can't be computed per fold.
Each fold gives an out-of-fold score for its subject, and one AUC is computed
over all of them at the end. Scaling lives inside the model (use a Pipeline) so
it is fitted on the training subjects only.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


def _score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def loso_scores(X, y, groups, model):
    """Out-of-fold score for every sample; each subject is scored by a model that never saw them."""
    X, y, groups = np.asarray(X, dtype=float), np.asarray(y), np.asarray(groups)
    scores = np.full(len(y), np.nan)
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        assert not set(groups[train]) & set(groups[test]), "subject appears in train and test"
        fitted = clone(model).fit(X[train], y[train])
        scores[test] = _score(fitted, X[test])
    return scores


def loso_auc(X, y, groups, model):
    """Return AUC under leave-one-subject-out CV."""
    return roc_auc_score(y, loso_scores(X, y, groups, model))


def bootstrap_auc_ci(y, scores, n_boot=2000, seed=0):
    """95% interval for AUC, resampling subjects with replacement."""
    y, scores = np.asarray(y), np.asarray(scores)
    rng = np.random.default_rng(seed)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], scores[idx]))
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def train_test_auc(X_train, y_train, X_test, y_test, model):
    """Fit on one cohort, score the other. Used for the cross-study check."""
    fitted = clone(model).fit(np.asarray(X_train, dtype=float), y_train)
    return roc_auc_score(y_test, _score(fitted, np.asarray(X_test, dtype=float)))
