"""The :mod:`salt.learn.cross_validation` module provides classes to handle cross-validation."""

from six.moves import range
import numpy as np


class CrossValidationGroup(object):
    def __init__(self, learner, parameters, dataset, folds=1):
        self.learner = learner
        self.parameters = parameters
        self.dataset = dataset
        self.folds = folds
        self.fold_labels = [None] * self.folds
        self._labels = None

    def __str__(self):
        return "[X-VAL GROUP for {0}]".format(self.learner.__name__)

    def create_folds(self):
        fold_list = []
        if self.folds == 1:
            # uses 2/3 - 1/3 by default for one fold (no cross-validation)
            # TODO: define right approach
            # TODO: Shuffle?
            testing_set, training_set = self.dataset.get_slice(1, 3)
            fold = CrossValidationFold(self.learner, self.parameters, training_set, testing_set)
            fold_list.append(fold)
        else:
            for i in range(self.folds):
                testing_set, training_set = self.dataset.get_slice(i + 1, self.folds)
                fold = CrossValidationFold(self.learner, self.parameters,
                                           training_set, testing_set)
                fold_list.append(fold)

        return fold_list

    def _get_labels(self):
        if self._labels is None:
            self._labels = np.hstack(self.fold_labels)
        return self._labels

    labels = property(_get_labels)


class CrossValidationFold(object):
    def __init__(self, learner, parameters, training_set, testing_set):
        self.learner = learner
        self.parameters = parameters
        self.training_set = training_set
        self.testing_set = testing_set
