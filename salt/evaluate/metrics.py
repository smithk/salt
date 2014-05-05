"""
The :mod:`salt.evaluate.metrics` module implements metrics to evaluate the results
of a learning task.
"""

from ..learn.classifiers import BaselineClassifier
from ..learn.regressors import BaselineRegressor

import numpy as np
from sklearn.metrics import (accuracy_score, fbeta_score, jaccard_similarity_score,
                             precision_score, recall_score, roc_auc_score, matthews_corrcoef,
                             average_precision_score, confusion_matrix as sk_confusion_matrix,
                             roc_curve as sk_roc_curve, precision_recall_curve,
                             mean_absolute_error, mean_squared_error, r2_score as sk_r2_score,
                             explained_variance_score)
from ..utils.arrays import proba_to_array, get_custom_confusion_matrix
import warnings


class BaseMetrics():
    def _get_score(self):
        raise NotImplementedError

    score = property(lambda self: self._get_score())

    # ======= Static and comparison methods =======

    @staticmethod
    def standardize_value(worst, best, value):
        assert worst != best
        return (1.0 * value - worst) / (1.0 * best - worst)

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


class RegressionMetrics(BaseMetrics):
    def __init__(self, expected, predicted, standardize=True, baseline=None):
        self._expected = expected
        self._predicted = predicted

        self._score = None
        self.standardize = standardize
        self.baseline = baseline

        self._explained_var = None
        self._mean_abs_error = None
        self._mean_sq_error = None
        self._r2_score = None
        self.weights = {'explained_var': 0.0,
                        'mean_abs_err': 1.0,
                        'mean_sq_err': 0.0,
                        'r2': 0.0}

    def __str__(self):
        return "Metrics:\nExplained variance: {0}\nMean_abs_err: {1}\nmean_sq_err: {2}\nr2: {3}\nSCORE:{4}".format(
            self.explained_var, self.mean_abs_error, self.mean_sq_error, self.r2_score, self.score)

    def _get_explained_var(self):
        if self._explained_var is None:
            self._explained_var = explained_variance_score(self._expected, self._predicted)
        return self._explained_var

    def _get_mean_abs_err(self):
        if self._mean_abs_error is None:
            computed_error = mean_absolute_error(self._expected, self._predicted)
            if self.standardize:
                self._mean_abs_error = BaseMetrics.standardize_value(self.baseline.mean_abs_error, 0, computed_error)
            else:
                self._mean_abs_error = computed_error
        return self._mean_abs_error

    def _get_mean_sq_err(self):
        if self._mean_sq_error is None:
            computed_error = mean_squared_error(self._expected, self._predicted)
            if self.standardize:
                self._mean_sq_error = BaseMetrics.standardize_value(self.baseline.mean_sq_error, 0, computed_error)
            else:
                self._mean_sq_error = computed_error
        return self._mean_sq_error

    def _get_r2_score(self):
        if self._r2_score is None:
            self._r2_score = sk_r2_score(self._expected, self._predicted)
        return self._r2_score

    explained_var = property(_get_explained_var)
    mean_abs_error = property(_get_mean_abs_err)
    mean_sq_error = property(_get_mean_sq_err)
    r2_score = property(_get_r2_score)

    def _get_score(self):
        if self._score is None:
            self._score = (self.explained_var * self.weights['explained_var'] +
                           self.mean_abs_error * self.weights['mean_abs_err'] +
                           self.mean_sq_error * self.weights['mean_sq_err'] +
                           self.r2_score * self.weights['r2']) / sum(self.weights.values())

        return self._score


class ClassificationMetrics(BaseMetrics):
    def __init__(self, expected=None, predicted=None, classes=None, standardize=True, baseline=None):
        self._classes = classes
        self._expected = None if expected is None else expected.astype(int)
        self.standardize = standardize
        self.baseline = baseline
        #self.predicted = proba_to_array(predicted)
        self._predicted = predicted
        self._predicted_class = None if predicted is None else proba_to_array(predicted)

        self.weights = {
            'accuracy': 1.0,
            'fscore': 1.0,
            'matthews': 1.0,
            'roc_auc': 0.0,
            'pr_auc': 0.0,
            'mean_abs_err': 0.0,
            'mean_sq_err': 0.0,
            'r2': 0.0,
        }

        self._total = None
        self._correct = None
        self._incorrect = None
        self._accuracy = None
        self._fscore = None
        self._precision = None
        self._recall = None
        self._roc_auc = None
        self._pr_auc = None
        self._matthews_corrcoef = None
        self._custom_confusion_accuracy = None
        self._custom_confusion_mask = None if self._expected is None else np.eye(len(np.unique(self._expected)))

        self._mean_abs_error = None
        self._mean_sq_error = None
        self._r2_score = None

        self._confusion_matrix = None
        self._roc_curve = None
        self._pr_curve = None

        self._score = None

        self._class_weights = None if self._classes is None else np.bincount(self._expected.astype(int),
                                                                             minlength=len(self._classes)) / (1.0 * len(self._expected))

    def __str__(self):
        return "Metrics:\nAccuracy: {0}\nFScore: {1}\nMatthews CC: {2}\nroc_auc: {3}\npr_auc: {4}\nmean_abs_err: {5}\nmean_sq_err: {6}\nr2: {7}\nSCORE:{8}".format(
            self.accuracy, self.fscore, self.matthews, self.roc_auc, self.pr_auc, self.mean_abs_error, self.mean_sq_error, self.r2_score, self.score)

    # ======== Metrics ========

    def _get_total(self):
        if self._total is None:
            self._total = len(self._expected)
        return self._total

    def _get_correct(self):
        if self._correct is None:
            # normalize=False => number of correctly classified samples
            # normalize=True => ratio of correctly classified samples (accuracy)
            self._correct = accuracy_score(self._expected, self._predicted_class, normalize=False)
        return self._correct

    def _get_incorrect(self):
        if self._incorrect is None:
            self._incorrect = self.total - self.correct
        return self._incorrect

    def _get_accuracy(self):
        if self._accuracy is None:
            # normalize=False => number of correctly classified samples
            # normalize=True => ratio of correctly classified samples (accuracy)
            self._accuracy = accuracy_score(self._expected, self._predicted_class, normalize=True)
            if self.standardize:
                self._accuracy = BaseMetrics.standardize_value(self.baseline.accuracy, 1, self._accuracy)
        return self._accuracy

    def _get_fscore(self):
        if self._fscore is None:
            # beta=0: precision only; beta=1: precision, recall (same weight); beta=Inf: recall only
            # default averaging for fbeta_score: weighted
            beta = 1
            self._fscore = fbeta_score(self._expected, self._predicted_class, beta)
            if self.standardize:
                self._fscore = BaseMetrics.standardize_value(self.baseline.fscore, 1, self._fscore)
        return self._fscore

    def _get_jaccard(self):
        # Do not use. Jaccard is equivalent to accuracy_score for multiclass classification
        if self._jaccard is None:
            jaccard_value = jaccard_similarity_score(self._expected, self._predicted_class)
            if self.standardize:
                #self._jaccard = BaseMetrics.standardize_value(-1, 1, jaccard_value)
                self._jaccard = BaseMetrics.standardize_value(self.baseline.jaccard, 1, jaccard_value)
            else:
                self._jaccard = jaccard_value
        return self._jaccard

    def _get_precision(self):
        if self._precision is None:
            self._precision = precision_score(self._expected, self._predicted_class, average='weighted')
            if self.standardize:
                self._precision = BaseMetrics.standardize_value(self.baseline.precision, 1, self._precision)
        return self._precision

    def _get_recall(self):
        if self._recall is None:
            self._recall = recall_score(self._expected, self._predicted_class, average='weighted')
            if self.standardize:
                self._recall = BaseMetrics.standardize_value(self.baseline.recall, 1, self._recall)
        return self._recall

    def _get_roc_auc(self):
        if self._roc_auc is None:
            weighted_roc_auc = sum([self._class_weights[i] *
                                    roc_auc_score(self._expected == i, self._predicted[:, i])
                                    if self._class_weights[i] > 0 else 0
                                    for i in np.arange(len(self._class_weights))])
            #self._roc_auc = roc_auc_score(self._expected, self._predicted_class)
            self._roc_auc = weighted_roc_auc
            if self.standardize:
                self._roc_auc = BaseMetrics.standardize_value(self.baseline.roc_auc, 1, self._roc_auc)
        return self._roc_auc

    def _get_pr_auc(self):
        if self._pr_auc is None:
            weighted_pr_auc = sum([self._class_weights[i] *
                                   average_precision_score(self._expected == i, self._predicted[:, i])
                                   for i in np.arange(len(self._class_weights))])
            self._pr_auc = weighted_pr_auc
            if self.standardize:
                self._pr_auc = BaseMetrics.standardize_value(self.baseline.pr_auc, 1, self._pr_auc)
        return self._pr_auc

    def _get_matthews_correlation_coefficient(self):
        if self._matthews_corrcoef is None:
            # TODO: confirm if _predicted_class==i or only use slice where
            # _expected = i
            weighted_mcc = sum([self._class_weights[i] *
                                matthews_corrcoef((self._expected == i).astype(int), (self._predicted_class == i).astype(int))
                                if self._class_weights[i] > 0 else 0
                                for i in np.arange(len(self._class_weights))])
            if self.standardize:
                #self._matthews_corrcoef = BaseMetrics.standardize_value(-1, 1, weighted_mcc)
                self._matthews_corrcoef = BaseMetrics.standardize_value(self.baseline.matthews, 1, weighted_mcc)
            else:
                self._matthews_corrcoef = weighted_mcc
        return self._matthews_corrcoef

    def _get_custom_confusion_accuracy(self):
        if self._custom_confusion_accuracy is None:
            conf_matrix = sk_confusion_matrix(self._expected, self._predicted_class)
            custom_confusion_matrix = get_custom_confusion_matrix(conf_matrix, self._custom_confusion_mask)
            self._custom_confusion_accuracy = sum(np.diagonal(custom_confusion_matrix)) / (1.0 * sum(sum(custom_confusion_matrix)))
        return self._custom_confusion_accuracy

    def _get_mean_abs_error(self):
        if self._mean_abs_error is None:
            self._mean_abs_error = mean_absolute_error(self._expected, self._predicted_class)
            if self.standardize:
                self._mean_abs_error = BaseMetrics.standardize_value(self.baseline.mean_abs_error, 0, self._mean_abs_error)
        return self._mean_abs_error

    def _get_mean_sq_error(self):
        if self._mean_sq_error is None:
            # TODO: Compute baseline
            computed_error = mean_squared_error(self._expected, self._predicted_class)
            if self.standardize:
                self._mean_sq_error = BaseMetrics.standardize_value(self.baseline.mean_sq_error, 0, computed_error)
            else:
                self._mean_sq_error = computed_error
        return self._mean_sq_error

    def _get_r2_score(self):
        if self._r2_score is None:
            # TODO: find out how to use probabilities here
            self._r2_score = sk_r2_score(self._expected, self._predicted_class)
            if self.standardize:
                self._r2_score = BaseMetrics.standardize_value(self.baseline.r2_score, 1, self._r2_score)
        return self._r2_score

    def _get_confusion_matrix(self):
        if self._confusion_matrix is None:
            self._confusion_matrix = sk_confusion_matrix(self._expected, self._predicted_class)
        return self._confusion_matrix

    def _get_roc_curve(self):
        if self._roc_curve is None:
            self._roc_curve = sk_roc_curve(self._expected, self._predicted)
        return self._roc_curve

    def _get_precision_recall_curve(self):
        if self._pr_curve is None:
            self._pr_curve = precision_recall_curve(self._expected, self._predicted_class)
        return self._pr_curve

    total = property(_get_total)
    correct = property(_get_correct)
    incorrect = property(_get_incorrect)
    accuracy = property(_get_accuracy)
    fscore = property(_get_fscore)
    precision = property(_get_precision)
    recall = property(_get_recall)
    roc_auc = property(_get_roc_auc)
    pr_auc = property(_get_pr_auc)
    matthews = property(_get_matthews_correlation_coefficient)

    mean_abs_error = property(_get_mean_abs_error)
    mean_sq_error = property(_get_mean_sq_error)
    r2_score = property(_get_r2_score)

    # ======== Visualizations ======== #

    confusion_matrix = property(_get_confusion_matrix)
    roc_curve = property(_get_roc_curve)
    pr_curve = property(_get_precision_recall_curve)

    # ======== Performance Index (Score) ==========

    def _get_score(self):
        if self._score is None:
            if self._predicted is None:
                self._score = 0
            else:
                # TODO: Create class to standardize and weight metrics.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._score = (self.accuracy * self.weights['accuracy'] +
                                   self.fscore * self.weights['fscore'] +
                                   self.roc_auc * self.weights['roc_auc'] +
                                   self.pr_auc * self.weights['pr_auc'] +
                                   self.matthews * self.weights['matthews'] +
                                   self.mean_abs_error * self.weights['mean_abs_err'] +
                                   self.mean_sq_error * self.weights['mean_sq_err'] +
                                   self.r2_score * self.weights['r2']) / sum(self.weights.values())

        return self._score


def get_baseline_metrics(dataset):
    learner = BaselineRegressor() if dataset.is_regression else BaselineClassifier()
    learner.train(dataset.data, dataset.target)
    prediction = learner.predict(dataset.data)
    if dataset.is_regression:
        baseline_metrics = RegressionMetrics(dataset.target, prediction, standardize=False)
    else:
        baseline_metrics = ClassificationMetrics(dataset.target, prediction, dataset.target_names, standardize=False)
    return baseline_metrics
