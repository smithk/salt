"""
The :mod:`salt.parameters.base` module describes the base classes for description
of the parameter space.
"""

from sklearn.grid_search import ParameterGrid
from scipy import stats
import numpy as np
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

    def sample_configuration(self):
        configuration = {}
        for param_name, param_spec in iteritems(self.__dict__):
            if isinstance(param_spec, Parameter):
                if param_spec.sample_if is None:  # or param_spec.sample_if(configuration) is True:
                    value = param_spec.get_a_value(root_config=configuration)  # self)
                    configuration[param_name] = value
        for param_name, param_spec in iteritems(self.__dict__):
            if isinstance(param_spec, Parameter):
                if param_spec.sample_if is not None:
                    if param_spec.sample_if(configuration):
                    #for child in param_spec.children:
                    #    child_config = child.sample_configuration(configuration)
                    #    configuration.update(child_config)
                        value = param_spec.get_a_value(root_config=configuration)  # self)
                        configuration[param_name] = value
        return configuration


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


class ConfigurationSpace(object):
    def __init__(self, top_level_params):
        assert type(top_level_params) is dict
        self.params = top_level_params

    def get_configuration(self):
        configuration = {param.name: param.get_value() for param in self.params}
        return configuration

    def update_prior(self, configuration, score):
        pass


class Configuration(object):
    def __init__(self, parameter_values):
        assert type(parameter_values) is dict
        self.parameter_values = parameter_values


class Parameter(object):
    def __init__(self, name, prior, children=None, default=None, sample_if=None, valid_if=None, action_if_invalid='use_default', discretize=None):
        self.name = name
        if type(prior) is list:
            prior = UniformDist(prior)
        elif type(prior) is dict:
            prior = CategoricalDist(prior)
        self.prior = prior
        self.distribution = prior  # Learnt distribution
        self.children = children
        self.sample_if = sample_if
        self.valid_if = valid_if
        if action_if_invalid not in ('use_default', 'use_none', 'resample'):
            pass  # Report exception?
        self.action_if_invalid = action_if_invalid
        self.default = default
        self.discretize = discretize

    def get_a_value(self, root_config=None):
        # always returns a flattened dictionary
        value = self.distribution.get_sample(self.discretize)
        if self.valid_if is not None and not self.valid_if(value, root_config):
            if self.action_if_invalid == 'use_none':
                value = None
            elif self.action_if_invalid == 'use_default':
                value = self.default
            else:
                value = self.get_a_value(root_config)
        if type(value) is np.bool_:  # Conversion to python scalar
            value = value.item()
        # TODO Consider to separate discretization into a different function?
        return value

    def is_valid(self, value, root_config):
        valid = self.sample_if(root_config) if self.sample_if is not None else True
        return valid
        # valid = all((self.validate_rule(rule, root_config) for rule in self.sample_if))
        # return valid

    def validate_rule(self, rule, root_config):
        param_to_validate, operator, target_to_validate = rule
        value_to_validate = root_config[param_to_validate]
        validators = {'=': lambda val, target: val == target,
                      '!=': lambda val, target: val != target,
                      'is': lambda val, target: type(val) is target,
                      '>': lambda val, target: val > target,
                      '>=': lambda val, target: val >= target,
                      '<': lambda val, target: val < target,
                      '<=': lambda val, target: val <= target}
        validator = validators[operator]
        validation_value = validator(value_to_validate, target_to_validate)
        return validation_value


class CategoricalParam(Parameter):
    def __init__(self, name, prior, children=None):
        if type(prior) is list:
            prior = UniformDist(prior)
        elif type(prior) is dict:
            prior = CategoricalDist(prior)
        super(CategoricalParam, self).__init__(name, prior, children)


"""
def create_param_space():
    params = {'neighbors': DiscreteParam(prior=UniformDist(lower=1, upper=100)),
              'metric':    CategoricalParam(prior=UniformDist(categories=['euclidean', 'manhattan', 'chebyshev']),
                                            children={'p': ContinuousParam(prior=UniformDist(lower=3), validation_rule = ('metric', '=', 'minkowski'), failback_action='use_default', default=2)}),
              'algorithm': CategoricalParam(prior=CategoricalDist(categories={'brute': 0.3, 'ball_tree': 0.4, 'kd_tree': 0.3}),
                                            children={'leaf_size': DiscreteParam(prior=GaussianDist(mean=30, stdev=10), validation_rule = ('algorithm', '!=', 'brute'), failback_action='use_None', default=30)})
            }

    params = {'neighbors': DiscreteParam(prior=UniformDist(lower=1, upper=100)),
              'metric':    CategoricalParam(prior=UniformDist(categories=['euclidean', 'manhattan', 'chebyshev']),
                                            children={'p': ContinuousParam(prior=UniformDist(lower=3), sample_if = ('metric', '=', 'minkowski'), default=2)}),
              'algorithm': CategoricalParam(prior=CategoricalDist(categories={'brute': 0.3, 'ball_tree': 0.4, 'kd_tree': 0.3}),
                                            children={'leaf_size': DiscreteParam(prior=GaussianDist(mean=30, stdev=10), sample_if = ('algorithm', '!=', 'brute'), default=30)})
            }

    params = {'neighbors': DiscreteParam(UniformDist(lower=1, upper=100)),
              'metric':    CategoricalParam(['euclidean', 'manhattan', 'chebyshev']),
                                            children={'p': ContinuousParam(UniformDist(lower=3), sample_if = ('metric', '=', 'minkowski'), default=2)}),
              'algorithm': CategoricalParam({'brute': 0.3, 'ball_tree': 0.4, 'kd_tree': 0.3},
                                            children={'leaf_size': DiscreteParam(GaussianDist(mean=30, stdev=10), sample_if = ('algorithm', '!=', 'brute'), default=30)})
            }
"""


class Distribution(object):
    def __init__(self):
        pass

    def get_sample(self, discretize=None):
        pass

    @classmethod
    def load(self, settings, category_type='string', parse_none=False):
        # Loads a distribution specified by a dictionary
        distribution = None
        distribution_type = settings.get('distribution')
        categories = settings.get('categories')
        if categories:
            categories = [Distribution._parse(category, category_type, parse_none) for category in categories]
        if distribution_type.lower() == 'categorical':
            probabilities = settings.get('probabilities')
            if probabilities is None:
                distribution = CategoricalDist(categories)
            else:
                category_dist = {key: float(value) for (key, value) in zip(categories, probabilities)}
                distribution = CategoricalDist(category_dist)
        elif distribution_type.lower() == 'uniform':
            lower = settings.as_float('lower')
            upper = settings.as_float('upper')
            distribution = UniformDist(lower, upper, categories)
        elif distribution_type.lower() == 'loguniform':
            lower = settings.as_float('lower')
            upper = settings.as_float('upper')
            distribution = LogUniformDist(lower, upper, categories)
        elif distribution_type.lower() in ('normal', 'gaussian'):
            mean = settings.as_float('mean')
            stdev = settings.as_float('stdev')
            distribution = NormalDist(mean, stdev, categories)
        elif distribution_type.lower() == 'lognormal':
            mean = settings.as_float('mean')
            stdev = settings.as_float('stdev')
            distribution = LogNormalDist(mean, stdev, categories)
        else:
            pass  # TODO Report invalid distribution specification
        return distribution

    @classmethod
    def _parse(self, value, datatype='string', parse_none=False):
        value = str(value)
        if parse_none and value.lower() == 'none':
            return None
        elif datatype == 'float':
            return float(value)
        elif datatype == 'int':
            return int(value)
        elif datatype == 'bool':
            return value.lower() == 'true'
        else:
            return value


class UniformDist(Distribution):
    def __init__(self, lower=0.0, upper=1.0, categories=None):
        self.lower, self.upper = (lower, upper) if lower <= upper else (upper, lower)
        self.categories = categories

    def get_sample(self, discretize=None):
        if self.categories is None:
            if discretize == 'int':
                value = np.random.rint(self.lower, self.upper + 1)
            elif discretize == 'round':
                value = int(np.rint(np.random.uniform(self.lower, self.upper)))
            else:
                value = np.random.uniform(self.lower, self.upper)
        else:
            value = np.random.choice(self.categories)
        #if self.categories is not None:
        #    bin_boundaries = np.linspace(self.lower, self.upper, len(self.categories) + 1)
        #    n_bin = np.digitize(value, bin_boundaries)
        #    value = self.categories[n_bin - 1]
        return value


class LogUniformDist(UniformDist):
    def __init__(self, lower=1000 * np.finfo(np.float32).eps, upper=100, categories=None):
        self.lower, self.upper = (lower, upper) if lower <= upper else (upper, lower)
        self.lower = max(self.lower, 10 * np.finfo(np.float32).eps)
        self.lower = np.log(self.lower)
        self.upper = np.log(self.upper)
        self.categories = categories

    def get_sample(self, discretize=None):
        if self.categories is None:
            if discretize == 'int':
                raise NotImplementedError
            elif discretize == 'round':
                value = np.exp(np.random.uniform(self.lower, self.upper))
                value = int(np.rint(value))
            else:
                value = np.exp(np.random.uniform(self.lower, self.upper))
        else:
            value = np.random.choice(self.categories)  # TODO: correct this
        return value


class NormalDist(Distribution):
    def __init__(self, mean=0, stdev=1, categories=None):
        self.mean = mean
        self.stdev = stdev
        self.categories = categories

    variance = property(lambda self: self.stdev ** 2)

    def get_sample(self, discretize=None):
        value = np.random.normal(self.mean, self.stdev)
        if self.categories is not None:
            bin_boundaries = stats.norm(loc=self.mean, scale=self.stdev).ppf(np.linspace(0, 1, len(self.categories), False)[1:])
            bin_index = np.digitize(value, bin_boundaries)
            value = self.categories[bin_index]
        if discretize == 'int':
            raise NotImplementedError
        elif discretize == 'round':
            value = int(np.rint(value))

        return value


class LogNormalDist(NormalDist):
    def get_sample(self, discretize=None):
        value = np.random.lognormal(self.mean, self.stdev)
        if self.categories is not None:
            bin_boundaries = stats.lognorm(self.stdev, loc=self.mean).ppf(np.linspace(0, 1, len(self.categories), False)[1:])
            bin_index = np.digitize([value], bin_boundaries)
            value = self.categories[bin_index[0]]
        if discretize == 'int':
            raise NotImplementedError
        elif discretize == 'round':
            value = int(np.rint(value))
        return value


class CategoricalDist(Distribution):
    def __init__(self, categories):
        # Weights can be specified (dictionary) or assumed uniform (list)
        if type(categories) is list:
            uniform_prob = 1. / len(categories)
            categories = {category: uniform_prob for category in categories}
        assert type(categories) is dict
        self.categories = categories

    def get_sample(self, update=True):
        sample = np.random.choice(self.categories.keys(), p=self.categories.values())
        return sample
