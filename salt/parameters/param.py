#from sklearn.grid_search import ParameterGrid
##from scipy import stats
import numpy as np
#from ..sample.distributions import UniformDistribution
from six import iteritems
from collections import Mapping, OrderedDict
#import inspect
from copy import deepcopy
from sklearn.mixture import GMM


class Distribution(object):
    """Base class for distributions"""
    def __init__(self):
        super(Distribution, self).__init__()

    @classmethod
    def load(self, settings, category_type='string', parse_none=False):
        # Loads a distribution specified by a dictionary
        distribution = None
        distribution_type = settings.get('distribution')
        categories = settings.get('categories')
        if categories:
            categories = [Distribution._parse(category, category_type, parse_none) for category in categories]
        if distribution_type.lower() == 'categorical':
            pass  # TODO FIX THIS
            #probabilities = settings.get('probabilities')
            #if probabilities is None:
            #    distribution = CategoricalDist(categories)
            #else:
            #    category_dist = {key: float(value) for (key, value) in zip(categories, probabilities)}
            #    distribution = CategoricalDist(category_dist)
        elif distribution_type.lower() == 'uniform':
            lower = settings.as_float('lower')
            upper = settings.as_float('upper')
            distribution = Uniform(lower, upper)
        elif distribution_type.lower() == 'loguniform':
            lower = settings.as_float('lower')
            upper = settings.as_float('upper')
            distribution = LogUniform(lower, upper)
        elif distribution_type.lower() in ('normal', 'gaussian'):
            mean = settings.as_float('mean')
            stdev = settings.as_float('stdev')
            distribution = Normal(mean, stdev)
        elif distribution_type.lower() == 'lognormal':
            mean = settings.as_float('mean')
            stdev = settings.as_float('stdev')
            distribution = LogNormal(mean, stdev)
        elif distribution_type.lower() == 'gmm':
            means = settings.get_list('means')
            variances = settings.get_list('variances')
            weights = settings.get_list('weights')
            distribution = GaussianMixture(means, variances, weights)
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

    def dump(self):
        """Dump the contents of the distribution into a dictionary-like object."""
        return {}

    def __str__(self):
        return "[BaseDistribution]"


class Uniform(Distribution):
    """Uniform distribution"""
    def __init__(self, lower=0, upper=1):
        super(Uniform, self).__init__()
        lower = float(lower)
        upper = float(upper)
        self.lower, self.upper = (lower, upper) if lower <= upper else (upper, lower)

    def get_sample(self, discretize=None):
        '''
        if discretize == 'int':
            value = np.random.randint(self.lower, self.upper + 1, size=None)
            value = int(value)
        elif discretize == 'round':
            value = int(np.rint(np.random.uniform(self.lower, self.upper, size=None)))
        else:
            value = np.random.uniform(self.lower, self.upper, size=None)
            value = float(value)
        return value
        '''
        if discretize is True:
            """
            # Already implemented in numpy
            try:
                lower = np.ceil(float(self.lower)) - 0.5
            except TypeError:
                print(self.lower, type(self.lower))
            upper = np.floor(float(self.upper)) + 0.5
            value = np.random.uniform(lower, upper, size=None)
            value = int(np.round(value))
            """
            if self.lower < self.upper:
                value = np.random.random_integers(self.lower, self.upper)
            else:
                value = self.upper
        else:
            value = np.random.uniform(self.lower, self.upper, size=None)
            value = float(value)
        return value

    def dump(self):
        """Dump the contents of the distribution into a dictionary-like object."""
        dist = OrderedDict()
        dist['distribution'] = 'uniform'
        dist['lower'] = self.lower
        dist['upper'] = self.upper
        return dist

    def __str__(self):
        return 'Uniform({0} to {1})'.format(self.lower, self.upper)


class LogUniform(Distribution):
    """LogUniform distribution"""
    def __init__(self, lower=-10, upper=10):
        super(LogUniform, self).__init__()
        lower = float(lower)
        upper = float(upper)
        self.lower, self.upper = (lower, upper) if lower <= upper else (upper, lower)
        if lower < np.log(np.finfo(np.float).eps):
            raise ValueError("exp(lower bound) must be greater than machine epsilon")

    def get_sample(self, discretize=None):
        value = np.exp(np.random.uniform(self.lower, self.upper, size=None))
        if discretize is True:
            value = int(np.rint(value))  # TODO Check if further processing needed
        else:
            value = float(value)
        return value

    def dump(self):
        """Dump the contents of the distribution into a dictionary-like object."""
        dist = OrderedDict()
        dist['distribution'] = 'loguniform'
        dist['lower'] = self.lower
        dist['upper'] = self.upper
        return dist

    def __str__(self):
        return 'LogUniform({0} to {1})'.format(self.lower, self.upper)


class Normal(Distribution):
    """Normal (Gaussian) distribution"""
    def __init__(self, mean=0, stdev=1):
        self.mean = float(mean)
        self.stdev = float(stdev)

    variance = property(lambda self: self.stdev ** 2)

    def get_sample(self, discretize=None, num_points=None):
        value = np.random.normal(loc=self.mean, scale=self.stdev, size=num_points)
        if discretize is True:
            if num_points is None:
                value = int(np.rint(value))
            else:
                value = value.astype(int)
        else:
            if num_points is None:
                value = float(value)
            else:
                value = value.astype(float)
        return value

    def dump(self):
        """Dump the contents of the distribution into a dictionary-like object."""
        dist = OrderedDict()
        dist['distribution'] = 'normal'
        dist['mean'] = self.mean
        dist['stdev'] = self.stdev
        return dist

    def __str__(self):
        return 'Normal({0}, {1})'.format(self.mean, self.stdev)


class LogNormal(Distribution):
    """LogNormal distribution"""
    def __init__(self, mean=0, stdev=1):
        super(LogNormal, self).__init__()
        self.mean = float(mean)
        self.stdev = float(stdev)

    def get_sample(self, discretize=None):
        value = np.random.lognormal(mean=self.mean, sigma=self.stdev, size=None)
        if discretize is True:
            value = int(np.round(value))
        else:
            value = float(value)
        return value

    def dump(self):
        """Dump the contents of the distribution into a dictionary-like object."""
        dist = OrderedDict()
        dist['distribution'] = 'lognormal'
        dist['mean'] = self.mean
        dist['stdev'] = self.stdev
        return dist

    def __str__(self):
        return 'LogNormal(mu={0}, sigma={1})'.format(self.mean, self.stdev)


class GaussianMixture(Distribution):
    def __init__(self, means=None, stdevs=None, weights=None):
        super(GaussianMixture, self).__init__()
        if means is not None:
            self.gmm = GMM(n_components=len(means))
            self.means = np.atleast_2d(means).T
            self.variances = np.atleast_2d(stdevs).T ** 2
            self.weights = np.atleast_2d(weights)
            self.gmm.means_ = self.means
            self.gmm._set_covars(self.variances)
            self.gmm.weights_ = self.weights
        else:
            self.gmm = GMM(n_components=1)
        self.data = []

    def reset(self):
        self.gmm = None

    def get_sample(self, num_points=1):
        sampled_values = self.gmm.sample(num_points)
        return sampled_values

    def eval(self, x_points):
        values = self.gmm.score(x_points)
        return values


    @classmethod
    def belongs(self, sample, gmm):
        pass

    def fit(self, data):
        self.gmm.fit(data)
        self.means = self.gmm.means_
        self.variances = self.gmm.covars_
        self.weights = self.gmm.weights_

    def combine(self, sample):
        if type(sample) is list:
            self.data += sample
        else:
            self.data += [sample]
        if len(self.data) <= self.gmm.n_components:
            return
        else:
            self.gmm.fit(self.data)
        new_gmm = GMM(n_components=self.gmm.n_components + 1)
        new_gmm.fit(self.data)
        new_aic = new_gmm.aic(self.data)
        curr_aic = self.gmm.aic(self.data)

        relative_likelihood = np.exp((new_aic - curr_aic) / 2)
        if relative_likelihood < 2e-5:
            print("updating model to distribution with {0} components".format(new_gmm.n_components))
            self.gmm = new_gmm
        elif self.gmm.n_components > 1:
            simpler_gmm = GMM(n_components=self.gmm.n_components - 1)
            simpler_gmm.fit(self.data)
            if simpler_gmm.aic(self.data) < curr_aic:
                self.gmm = simpler_gmm
            #self.means = self.gmm.means_
            #self.variances = self.gmm._get_covars()
            #self.weights = self.gmm.weights_

    def simplify(self):
        pass

    def get_plot_data(self, start, end, num_points):
        x_points = np.linspace(start, end + 1, num_points)
        y_points = np.exp(self.gmm.score(x_points))
        return x_points, y_points

    def dump(self):
        dist = OrderedDict()
        dist['distribution'] = 'gmm'
        dist['means'] = self.means
        dist['variances'] = self.variances
        dist['weights'] = self.weights
        return dist



class ParameterSpace(object):
    """Describes the parameter space"""
    def __init__(self, name='default', categorical_params=None, numerical_params=None):
        super(ParameterSpace, self).__init__()
        self.categorical_params = categorical_params or {}
        self.numerical_params = numerical_params or {}
        self.name = name

    def sample_configuration(self):
        """Returns a parameter space configuration"""
        configuration = {}
        for parameter in self.categorical_params.values():
            partial_configuration = parameter.sample_configuration()
            configuration.update(partial_configuration)
        for parameter in self.numerical_params.values():
            configuration[parameter.name] = parameter.sample_configuration(configuration)
        ##    numerical_parameter = self.find_numerical_parameter(configuration)
        #    configuration[numerical_parameter.name] = numerical_parameter.sample_configuration()
        return configuration

    def get_default(self):
        configuration = {}
        for parameter in self.categorical_params.values():
            partial_configuration = parameter.get_default()
            configuration.update(partial_configuration)
        for parameter in self.numerical_params.values():
            configuration[parameter.name] = parameter.get_default(configuration)
        return configuration

    def __repr__(self):
        depth = 0
        param_string = ' ' * 4 * depth + 'categorical:\n'
        for name, cat_param in iteritems(self.categorical_params):
            cat_string = str(cat_param)
            param_string += ' ' * 4 * depth + '    ' + name + ': ' + ('\n' + cat_string) \
                if len(cat_string) > 0 else '' + '\n'
        param_string += ' ' * 4 * depth + 'Numerical:\n'
        for name, num_param in iteritems(self.numerical_params):
            param_string += ' ' * 4 * depth + '    ' + name + ': ' + str(num_param) + '\n'
        return param_string

    def __str__(self):
        return self.__repr__()

    def __setitem__(self, key, val):
        if type(val) is CategoricalParameter:
            self.categorical_params[key] = val
        elif type(val) is NumericalParameter:
            self.numerical_params[key] = val
        else:
            dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        if key in self.categorical_params:
            return self.categorical_params[key]
        elif key in self.numerical_params:
            return self.numerical_params[key]
        else:
            return self[key]

    @classmethod
    def _get_keyword(self, string):
        if type(string) is str and string[0] == '$':
            value = {'false': False, 'true': True, 'none': None}[string[1:].lower()]
        else:
            value = string
        return value

    @classmethod
    def load(self, param_spec):
        """Loads a parameter space."""
        categorical_params = param_spec.get("CategoricalParams", {})
        numerical_params = param_spec.get("NumericalParams", {})

        param_space = ParameterSpace()

        for name, param in iteritems(categorical_params):
            default = param.get('default', None)
            default = self._get_keyword(default)
            categories = {self._get_keyword(cat_name): float(cat_val.get('weight', 0))
                          for cat_name, cat_val in iteritems(param)
                          if isinstance(cat_val, Mapping)}
            parameter = CategoricalParameter(name, categories, default)
            for category in categories.keys():
                category_index = ('$' if type(category) is not str else '') + str(category)
                if param[category_index].get('CategoricalParams', False) or param[category_index].get('NumericalParams', False):
                    parameter.categories[category] = self.load(param[category_index])
            param_space[name] = parameter

        for name, param in iteritems(numerical_params):
            discretize = self._get_keyword(param.get('discretize', '_'))
            default = param.get('default')  # TODO: parse types
            default = self._get_keyword(default)
            if type(default) is str:
                default = int(float(default)) if discretize else float(default)  # Check for errors in config file?
            distribution = param.get('distribution', '').lower()
            valid_if = param.get('valid_if')
            when_invalid = param.get('when_invalid')
            if distribution == 'uniform':
                prior = Uniform(lower=param.get('lower', None), upper=param.get('upper', None))
            elif distribution == 'loguniform':
                prior = LogUniform(lower=param.get('lower', None), upper=param.get('upper', None))
            elif distribution in ('normal', 'gaussian'):
                prior = Normal(mean=param.get('mean', None), stdev=param.get('stdev', None))
            elif distribution == 'lognormal':
                prior = LogNormal(mean=param.get('mean', None), stdev=param.get('stdev', None))
            else:
                raise ValueError("Invalid distribution type", distribution)
            param_space[name] = NumericalParameter(name, prior=prior, default=default, discretize=discretize, valid_if=valid_if, when_invalid=when_invalid)

        return param_space

    def dump(self):
        space = OrderedDict()
        if len(self.categorical_params) > 0:
            space['CategoricalParams'] = OrderedDict()
            for name, value in iteritems(self.categorical_params):
                param = OrderedDict()
                param['default'] = value.default if type(value.default) is str else ('$' + str(value.default))
                for cat_name, cat_value in iteritems(value.categories):
                    if cat_name is None or type(cat_name) is bool:
                        temp_cat_name = '$' + str(cat_name)
                    else:
                        temp_cat_name = cat_name
                    if isinstance(cat_value, ParameterSpace):
                        param[temp_cat_name] = cat_value.dump()
                        param[temp_cat_name]['weight'] = value.distribution[cat_name]
                    else:
                        param[temp_cat_name] = cat_value
                space['CategoricalParams'][str(name)] = param
        if len(self.numerical_params) > 0:
            space['NumericalParams'] = OrderedDict()
            for name, value in iteritems(self.numerical_params):
                space['NumericalParams'][str(name)] = value.dump()
        return space

    @classmethod
    def get_numerical_space(self, param_space, configuration):
        assert isinstance(configuration, Mapping)
        numerical_space = {}  # copy instead of reference to param_space field
        numerical_space.update(param_space.numerical_params)
        for cat in param_space.categorical_params.values():
            numerical_space.update(self.get_numerical_space(cat.categories[configuration[cat.name]], configuration))
        return numerical_space

    @classmethod
    def get_cat_signature(self, param_space, configuration):
        try:
            assert isinstance(configuration, Mapping)
            categorical_space = {name: configuration[name] for name in param_space.categorical_params.keys()}
            for cat in param_space.categorical_params.values():
                categorical_space.update(self.get_cat_signature(cat.categories[configuration[cat.name]], configuration))
            ordered_categorical_space = OrderedDict(sorted(categorical_space.items()))
            return ordered_categorical_space
        except KeyError as key_exc:
            print("An error happened when attempting to retrieve the signature"
                  "for\n\t{0}\non\n\t{1}".format(configuration, param_space.categorical_params))


class Parameter(object):
    """Base class for parameters"""
    def __init__(self, name):
        super(Parameter, self).__init__()
        self.name = name


class CategoricalParameter(Parameter):
    """Categorical parameter definition"""
    def __init__(self, name, categories, default):
        super(CategoricalParameter, self).__init__(name)
        if type(categories) is list:
            uniform_prob = 1. / len(categories)
            categories = {category: uniform_prob for category in categories}
        assert type(categories) is dict
        self.distribution = categories
        self.categories = {category: ParameterSpace(name=category) for category in categories.keys()}
        self.default = default

    def sample_configuration(self):
        configuration = {}
        if len(self.distribution) > 0:
            value = np.random.choice(list(self.distribution.keys()), p=list(self.distribution.values()))
            if type(value) is np.bool_:  # Conversion to python scalar
                value = value.item()
            configuration = self.categories[value].sample_configuration()
            configuration[self.name] = value
        return configuration

    def get_default(self):
        configuration = {}
        if len(self.distribution) > 0:
            value = self.default
            if type(value) is np.bool_:  # Conversion to python scalar
                value = value.item()
            if value in self.categories:
                configuration = self.categories[value].get_default()
            if configuration:
                configuration[self.name] = value
            else:
                configuration = {}
        return configuration

    def __repr__(self):
        param_string = ''
        for category_name in self.distribution.keys():
            #param_string += str(category_name) + '\n'
            param_string += '    (' + str(category_name) + '\n' + str(self.categories[category_name]) + ')\n'
        return param_string

    def __str__(self):
        return self.__repr__()


class NumericalParameter(Parameter):
    """Numerical parameter definition"""
    def __init__(self, name, default, prior=Uniform(0, 1), valid_if=None, when_invalid='use_default', discretize=None):
        super(NumericalParameter, self).__init__(name)
        self.prior = prior
        self.distribution = deepcopy(prior)  # (learnt distribution)
        self.valid_if = valid_if
        if self.valid_if is not None:
            if when_invalid not in ('use_default', 'use_none', 'resample'):
                raise ValueError
        self.when_invalid = when_invalid
        self.default = default
        self.discretize = discretize

    def dump(self):
        """Dump the contents of the parameter into a dictionary-like object."""
        param = self.prior.dump()
        param['default'] = self.default if type(self.default) in (float, int) else ('$' + str(self.default))
        if self.discretize is not None:
            param['discretize'] = '$' + str(self.discretize)
        if self.valid_if is not None:
            param['valid_if'] = self.valid_if
            param['when_invalid'] = self.when_invalid
        return param

    def __repr__(self):
        return self.__class__.__name__  # "NumericalParameter"

    def sample_configuration(self, current_config):
        try:
            value = self.distribution.get_sample(self.discretize)
            if self.valid_if is not None:
                validation = eval("lambda value, params: " + self.valid_if)
                if not validation(value, current_config):
                    if self.when_invalid == 'use_none':
                        value = None
                    elif self.when_invalid == 'use_default':
                        value = self.default
                    else:
                        value = self.sample_configuration(current_config)  # TODO Control max number of attempts?
            if type(value) is np.bool_:
                value = value.item()
            return value
        except AttributeError:
            print(self.distribution)

    def get_default(self, current_config):
        return self.default


# For testing purposes
if __name__ == '__main__':
    from os import system
    system('clear')
    param_space = ParameterSpace()
    # Categorical parameters
    param_space['penalty'] = CategoricalParameter('penalty', {'l1': 0.5, 'l2': 0.5}, default='l2')
    param_space['penalty'].categories['l2']['dual'] = CategoricalParameter('dual', {None: 0.5, 'auto': 0.5}, default=None)
    param_space['fit_intercept'] = CategoricalParameter('fit_intercept', {True: 0.5, False: 0.5}, default=False)
    param_space['fit_intercept'].categories[True]['intercept_scaling'] = NumericalParameter('intercept_scaling', prior=Uniform(0, 1), default=0)
    param_space['class_weight'] = CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)

    # Numerical parameters
    param_space['C'] = NumericalParameter('C', prior=LogUniform(0, 10), default=1, valid_if="params['class_weight'] is None")
    param_space['tol'] = NumericalParameter('tol', prior=LogNormal(mean=0, stdev=1), default=0.0001)

    #print(param_space)
    #print(param_space['penalty'].categories['l2'])
    sample = param_space.sample_configuration()
    num_params = ParameterSpace.get_numerical_space(param_space, sample)
    print(sample)
    print(num_params)
    print(ParameterSpace.get_cat_signature(param_space, sample))
    #print(param_space.dump())
