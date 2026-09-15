"""Leave-one-subject-out evaluation.

Why LOSO: with ~85 subjects, a random split puts the same person's keystrokes
in both train and test. The model then recognises the person, not the disease,
and AUC is inflated. Use sklearn.model_selection.LeaveOneGroupOut with
subject_id as the group.
"""


def loso_auc(X, y, groups, model):
    """Return AUC under leave-one-subject-out CV."""
    raise NotImplementedError
