"""The :mod:`salt.sample.distributions` module describes different probability distributions."""

from sklearn.grid_search import ParameterGrid
import numpy as np


class BaseDistribution(object):
    def discretize(num_points):
        raise NotImplementedError


class UniformDistribution(BaseDistribution):
    def __init__(self, lower=0, upper=1):
        if lower > upper:
            lower, upper = upper, lower
        self.lower = lower
        self.upper = upper

    def discretize(self, num_points):
        return np.linspace(self.lower, self.upper, num_points)
