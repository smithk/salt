import numpy as np


def array_to_proba(predicted, min_columns=None):
    proba = np.vstack([predicted == label for label in np.unique(predicted)]).T.astype(float)
    if min_columns and proba.shape[1] < min_columns:
        extra_columns = np.zeros((len(proba), min_columns - proba.shape[1]))
        proba = np.hstack([proba, extra_columns])
    return proba


def proba_to_array(proba, sample=True):
    if sample:
        return np.array([np.random.choice(len(row), p=row) for row in proba])
    else:
        return np.array([np.argmax(row) for row in proba])


def get_custom_confusion_matrix(confusion_matrix, new_confusion_mask):
    assert new_confusion_mask.ndim == 2
    assert new_confusion_mask.shape[0] == new_confusion_mask.shape[1]
    reverse_mask = -new_confusion_mask + 1
    new_confusion_matrix = np.diag([sum(confusion_matrix[i] * new_confusion_mask[i])
                                    for i in np.arange(confusion_matrix.shape[0])])
    new_confusion_matrix += np.vstack([confusion_matrix[i] * reverse_mask[i] for i in np.arange(confusion_matrix.shape[0])])
    return new_confusion_matrix
