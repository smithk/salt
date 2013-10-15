"""
The :mod:`salt.parameters.base` module describes the base classes for description
of the parameter space.
"""

from sklearn.grid_search import ParameterGrid
from ..sample.distributions import UniformDistribution
from six import iteritems


class ParameterSpace(dict):
    """Describe the parameter space."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

    def get_grid(self):
        grid = ParameterGrid({key: value.get_values() for key, value in iteritems(self)})
        return grid


class BaseParameter(object):
    """Base class for parameter objects."""
    def __init__(self):
        self.param_type = None

    def get_values(self):
        raise NotImplementedError


class ChoiceParameter(BaseParameter):  # TODO: consider to merge this with BaseParameter
    def __init__(self, choice_list):
        super(ChoiceParameter, self).__init__()
        self.choice_list = choice_list

    def get_values(self):
        return self.choice_list


class BooleanParameter(ChoiceParameter):
    def __init__(self):
        super(BooleanParameter, self).__init__(choice_list=[True, False])


class NumericRangeParameter(BaseParameter):
    def __init__(self, lower_bound, upper_bound, num_points, distribution=UniformDistribution):
        super(NumericRangeParameter, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_points = num_points
        self.distribution = distribution(lower_bound, upper_bound)

    def get_values(self):
        values = self.distribution.discretize(self.num_points)
        return values
