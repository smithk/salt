"""The :mod:`salt.optimize.base` module provides classes for optimization."""

from bisect import insort
from numpy.random import shuffle
from six.moves import range


class BaseOptimizer(object):
    def __init__(self, param_space):
        self.param_space = param_space
        self.evaluation_results = []
        self.evaluation_parameters = []

    def report_results(self, evaluation_results):
        insort(self.evaluation_results, evaluation_results)

    def get_next_configuration(self):
        raise NotImplementedError


class SequentialOptimizer(BaseOptimizer):
    def __init__(self, param_space):
        super(SequentialOptimizer, self).__init__(param_space)
        self.parameter_list = list(param_space.get_grid())
        param_space_size = len(self.parameter_list)
        self.indices = list(range(param_space_size - 1, -1, -1))

    def get_next_configuration(self):
        if not self.indices:
            return

        next_configuration = self.parameter_list[self.indices.pop()]
        return next_configuration


class RandomOptimizer(SequentialOptimizer):
    def __init__(self, param_space):
        super(RandomOptimizer, self).__init__(param_space)
        shuffle(self.indices)
