"""The :mod:`salt.evaluate.base` module provides general classes for learner evaluation."""


class EvaluationResults(object):
    def __init__(self, learner, parameters, metrics):
        self.learner = learner
        self.parameters = parameters
        self.metrics = metrics

    def __lt__(self, other):
        return self.metrics.score < other.metrics.score
