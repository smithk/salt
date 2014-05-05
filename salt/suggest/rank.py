"""The :mod:`salt.suggest.rank` module provides functionality for ranking learner suggestions."""
import numpy as np
from sklearn.cluster import k_means


def rank(suggestion_task_list, top=10):
    return [0] * top


def get_local_maxima(evaluation_list, k=10):
    if len(evaluation_list) <= k:
        return evaluation_list

    scores = np.array([np.mean(score) for score, runtime, config in evaluation_list])
    #scores = np.array([score for score, config in evaluation_list])
    means, labels, error = k_means(np.atleast_2d(scores).T, k)
    selected_models = []
    #for i in xrange(len(np.unique(means))):
    for i in np.unique(labels):
        indices = np.argwhere(labels == i)
        best = np.argmax(scores[indices])
        for index in indices[best]:
            selected_models.append(evaluation_list[index])
        # selected_means.append(scores[indices[best]])
    return selected_models
