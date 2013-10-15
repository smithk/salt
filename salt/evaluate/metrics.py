"""
The :mod:`salt.evaluate.metrics` module implements metrics to evaluate the results
of a learning task.
"""

import numpy as np


class Metrics(object):
    def __init__(self, expected, predicted):
        self.expected = expected
        self.predicted = predicted

    def _clean_metrics(self):
        self._total = None
        self._correct = None
        self._incorrect = None
        self._accuracy = None
        self._score = None

    def _set_expected(self, expected):
        self._clean_metrics()
        self._expected = expected

    def _set_predicted(self, predicted):
        self._clean_metrics()
        self._predicted = predicted

    # metrics

    def _get_total(self):
        if self._total is None:
            self._total = len(self.expected)
        return self._total

    def _get_correct(self):
        if self._correct is None:
            self._correct = np.sum(self.expected == self.predicted)
        return self._correct

    def _get_incorrect(self):
        if self._incorrect is None:
            self._incorrect = self.total - self.correct
        return self._incorrect

    def _get_accuracy(self):
        if self._accuracy is None:
            self._accuracy = self.correct / float(self.total)
        return self._accuracy

    def _get_score(self):
        if self._score is None:
            # TODO: Create class to standarize and weight metrics.
            self._score = self.accuracy
        return self._score

    # make metrics objects comparable
    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __eq__(self, other):
        return self.score == other.score

    def __ne__(self, other):
        return self.score != other.score

    @staticmethod
    def standarize(worst, best, value):
        assert worst != best
        return (1.0 * value - worst) / (best - worst)

    expected = property(lambda self: self._expected, _set_expected)
    predicted = property(lambda self: self._predicted, _set_predicted)

    total = property(_get_total)
    correct = property(_get_correct)
    incorrect = property(_get_incorrect)
    accuracy = property(_get_accuracy)

    score = property(_get_score)
