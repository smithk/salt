"""
The :mod:`salt.learn` subpackage implements the classes used to perform training and prediction
on classification, regression, and clustering problems.

.. todo::

    Register learners dynamically.

"""

from .classifiers import LinearSVMClassifier, GaussianNaiveBayesClassifier, KNNClassifier


AVAILABLE_CLASSIFIERS = {'linearsvm': LinearSVMClassifier,
                         'gaussiannb': GaussianNaiveBayesClassifier,
                         'knn': KNNClassifier,
                         }
DEFAULT_LEARNERS = [LinearSVMClassifier, GaussianNaiveBayesClassifier, KNNClassifier]
#DEFAULT_LEARNERS = [LinearSVMClassifier, KNNClassifier]
#DEFAULT_LEARNERS = [KNNClassifier]
