"""The :mod:`salt.learn.cross_validation` module provides classes to handle cross-validation."""
import numpy as np


class CrossValidationGroup(object):
    def __init__(self, learner, parameters, dataset, repetition, optimizer_id):
        self.learner = learner
        self.parameters = parameters
        self.dataset = dataset
        self.fold_labels = [None] * self.dataset.folds
        self._labels = None
        self.repetition = repetition
        self.runtime = 0
        self.optimizer_id = optimizer_id
        self.stats = None

    def __str__(self):
        return "[Cross Validation Group for {0}, repetition {1}. {2} folds]".format(self.learner.__name__, self.repetition, self.dataset.folds)

    def create_folds(self):
        fold_list = []
        #if self.dataset.folds == 1:
        #    #testing_set, training_set = self.dataset.get_slice(1, 3)
        #    testing_set, training_set = self.dataset.get_fold_data(1)
        #    fold = CrossValidationFold(self.learner, self.parameters, training_set, testing_set)
        #    fold_list.append(fold)
        #else:
        for i in xrange(self.dataset.folds):
            #testing_set, training_set = self.dataset._get_slice(i + 1, self.folds)
            testing_set, training_set = self.dataset.get_fold_data(self.repetition, i)
            fold = CrossValidationFold(self.learner, self.parameters,
                                       training_set, testing_set)
            fold_list.append(fold)

        return fold_list

    def _get_labels(self):
        try:
            if self._labels is None:
                # if fold_labels are coded as static labels instead of
                # probabilities, use hstack (i.e., if fold_labels are arrays
                # instead of matrices)
                if self.fold_labels[0].ndim == 1:
                    self._labels = np.hstack(self.fold_labels)
                else:
                    self._labels = np.vstack(self.fold_labels)
            return self._labels
        except Exception as exc:
            raise exc
            #print(len(self.fold_labels), [a.shape for a in self.fold_labels])

    labels = property(_get_labels)


class CrossValidationFold(object):
    def __init__(self, learner, parameters, training_set, testing_set):
        self.learner = learner
        self.parameters = parameters
        self.training_set = training_set
        self.testing_set = testing_set
