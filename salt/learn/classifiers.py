"""The :mod:`salt.learning.classifiers` module implements classes for classification."""

import warnings
from sklearn.linear_model import (LogisticRegression, SGDClassifier as SKSGDClassifier,
                                  PassiveAggressiveClassifier as SKPassiveAggressiveClassifier,
                                  RidgeClassifierCV, RidgeClassifier as SKRidgeClassifier)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import (KNeighborsClassifier,
                               RadiusNeighborsClassifier as SKRadiusNeighborsClassifier,
                               NearestCentroid)
from sklearn.tree import (DecisionTreeClassifier as SKDecisionTreeClassifier,
                          ExtraTreeClassifier as SKExtraTreeClassifier)
from sklearn.ensemble import (RandomForestClassifier as SKRandomForestClassifier,
                              ExtraTreesClassifier as SKExtraTreesClassifier,
                              GradientBoostingClassifier as SKGradientBoostingClassifier)
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.lda import LDA
from sklearn.qda import QDA
import numpy as np
from ..utils.arrays import array_to_proba
from .base import BaseLearner
from ..parameters import (ParameterSpace, ChoiceParameter, BooleanParameter, )
from ..parameters import Distribution, Parameter
from collections import OrderedDict


class BaseClassifier(BaseLearner):
    """Base class for classifiers. All classifiers should inherit from this class."""
    def __init__(self, **parameters):
        self._parameters = None
        self._classifier = None
        self.training_data = None
        self.training_target = None
        self.training_classes = None
        self.predict_data = None
        self._update_parameters(parameters)

    def _get_classifier(self):
        """Get a reference to a valid classifier object."""
        if not self._classifier:  # Ensures to provide a reference to a trainer
            self._classifier = self.create_classifier(**self.parameters)
        return self._classifier

    def _update_parameters(self, parameters):
        """Update learner parameters."""
        self._classifier = None  # forces to re-train on next use
        self._parameters = parameters

    parameters = property(lambda self: self._parameters, _update_parameters)
    classifier = property(lambda self: self._get_classifier())

    # --- Public methods ---

    def create_classifier(self, **parameters):
        """Call the classifier constructor for the specific class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        """
        raise NotImplementedError((self, parameters))

    def train(self, data, target):
        """Perform the training step on the given data.

        :param data: Training data.
        :param target: Training classes.
        """
        self.training_data = data
        self.training_target = target
        classifier = self.classifier
        self.training_classes = np.unique(target).astype(int)
        trained = classifier.fit(data, target.astype(int))
        return trained

    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        classifier = self.classifier
        prediction = classifier.predict_proba(data)
        # extended_prediction includes zero-filled columns for non-observed
        # classes. The returned array is sorted (first column corresponds to
        # class 0 and so on.
        extended_prediction = np.zeros((len(data), len(self.training_classes)))
        for index, class_id in enumerate(self.training_classes):
            extended_prediction[:, class_id] = prediction[:, index]
        return extended_prediction

    def get_learner_parameters(self, parameters):
        # DEPRECATED
        """Convert parameter space into parameters expected by the learner."""
        return (), {}

    def create_parameter_space(self, parameters=None, optimizer=None):
        return (), {}  # args, kwargs

    @classmethod
    def create_default_params(self):
        raise NotImplementedError


class BaselineClassifier(BaseClassifier):
    def train(self, data, target):
        self.training_classes = np.unique(target)
        self._weights = np.bincount(target.astype(int),
                                    minlength=len(self.training_classes)) / (1. * len(target))

    def predict(self, data):
        return np.repeat([self._weights], len(data), axis=0)


class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression classifier implementation."""

    def create_classifier(self, **parameters):
        """Creates the logistic regression classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Logistic Regression classifier.
        """
        classifier = LogisticRegression(**parameters)
        return classifier

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            penalty_options = OrderedDict()
            penalty_options['distribution'] = 'Categorical'
            penalty_options['categories'] = ['l1', 'l2']
            penalty_options['probabilities'] = [0.05, 0.95]

            dual_options = OrderedDict()
            dual_options['distribution'] = 'Categorical'
            dual_options['categories'] = [True, False]
            dual_options['probabilities'] = [0.25, 0.75]

            C_options = OrderedDict()
            C_options['distribution'] = 'LogUniform'
            C_options['lower'] = -10
            C_options['upper'] = 10

            fit_intercept_options = OrderedDict()
            fit_intercept_options['distribution'] = 'Categorical'
            fit_intercept_options['categories'] = [True, False]
            fit_intercept_options['probabilities'] = [0.5, 0.5]

            intercept_scaling_options = OrderedDict()
            intercept_scaling_options['distribution'] = 'Uniform'
            intercept_scaling_options['lower'] = 0.0
            intercept_scaling_options['upper'] = 5.0

            class_weight_options = OrderedDict()
            class_weight_options['distribution'] = 'Categorical'
            class_weight_options['categories'] = [None, 'auto']
            class_weight_options['probabilities'] = [0.5, 0.5]

            tolerance_options = OrderedDict()
            tolerance_options['distribution'] = 'LogNormal'
            tolerance_options['mean'] = 1e-4
            tolerance_options['stdev'] = 1.0

            learner_options = {'penalty': penalty_options,
                               'dual': dual_options,
                               'C': C_options,
                               'fit_intercept': fit_intercept_options,
                               'intercept_scaling': intercept_scaling_options,
                               'class_weight': class_weight_options,
                               'tolerance': tolerance_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Categorical parameters
        param_space['penalty'] = param.CategoricalParameter('penalty', {'l1': 0.05, 'l2': 0.95}, default='l2')
        param_space['penalty'].categories['l2']['dual'] = param.CategoricalParameter('dual', {True: 0.25, False: 0.75}, default=False)
        param_space['fit_intercept'] = param.CategoricalParameter('fit_intercept', {True: 0.5, False: 0.5}, default=False)
        param_space['fit_intercept'].categories[True]['intercept_scaling'] = param.NumericalParameter('intercept_scaling', prior=param.Uniform(0.0, 1.0), default=0.0)
        param_space['class_weight'] = param.CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)

        # Numerical parameters
        param_space['C'] = param.NumericalParameter('C', prior=param.LogUniform(-10.0, 10.0), default=0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=0, stdev=1), default=0.0001)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Categorical parameters

        class_weight_branch = param.CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)
        fit_intercept_branch = param.CategoricalParameter('fit_intercept', {True: 0.5, False: 0.5}, default=False)
        fit_intercept_branch.categories[True]['class_weight'] = class_weight_branch
        fit_intercept_branch.categories[False]['class_weight'] = class_weight_branch

        penalty_branch = param.CategoricalParameter('penalty', {'l1': 0.5, 'l2': 0.5}, default='l2')
        penalty_branch.categories['l1']['fit_intercept'] = fit_intercept_branch

        penalty_branch.categories['l2']['dual'] = param.CategoricalParameter('dual', {True: 0.5, False: 0.5}, default=False)
        penalty_branch.categories['l2']['dual'].categories[True]['fit_intercept'] = fit_intercept_branch
        penalty_branch.categories['l2']['dual'].categories[False]['fit_intercept'] = fit_intercept_branch
        param_space['penalty'] = penalty_branch

        # Numerical parameters
        param_space['C'] = param.NumericalParameter('C', prior=param.LogUniform(-10.0, 10.0), default=0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=0, stdev=1), default=0.0001)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['LogisticRegressionClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['LogisticRegressionClassifier'][optimizer]  # learner_settings['log_reg_classif']

        # Build priors
        penalty_prior = Distribution.load(learner_parameters['penalty'])
        dual_prior = Distribution.load(learner_parameters['dual'], category_type='bool')
        C_prior = Distribution.load(learner_parameters['C'])
        fit_intercept_prior = Distribution.load(learner_parameters['fit_intercept'], category_type='bool')
        intercept_scaling_prior = Distribution.load(learner_parameters['intercept_scaling'])
        class_weight_prior = Distribution.load(learner_parameters['class_weight'], parse_none=True)
        tolerance_prior = Distribution.load(learner_parameters['tolerance'])

        # Build parameters
        dual_param = Parameter('dual', prior=dual_prior, default=False,
                               sample_if=lambda params: params['penalty'] == 'l2')
        penalty_param = Parameter('penalty', prior=penalty_prior, default='l2')
        C_param = Parameter('C', prior=C_prior, default=1.0)
        intercept_scaling_param = Parameter('intercept_scaling', prior=intercept_scaling_prior,
                                            default=1.0, sample_if=lambda params: params['fit_intercept'])
        fit_intercept_param = Parameter('fit_intercept', prior=fit_intercept_prior, default=True)
        class_weight_param = Parameter('class_weight', prior=class_weight_prior, default=None)

        tolerance_param = Parameter('tol', prior=tolerance_prior, default=1e-4)

        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['penalty'] = penalty_param
        param_space['dual'] = dual_param
        param_space['tol'] = tolerance_param
        param_space['C'] = C_param
        param_space['fit_intercept'] = fit_intercept_param
        param_space['intercept_scaling'] = intercept_scaling_param
        param_space['class_weight'] = class_weight_param
        param_space['random_state'] = random_state_param
        return param_space


class SGDClassifier(BaseClassifier):
    """Stochastic Gradient Descent classifier implementation."""

    def create_classifier(self, **parameters):
        """Creates the stochastic gradient descent classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Stochastic Gradient Descent classifier.
        """
        classifier = SKSGDClassifier(**parameters)
        return classifier

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['SGDClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            loss_options = OrderedDict()
            loss_options['distribution'] = 'Categorical'
            loss_options['categories'] = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss',
                                          'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
            # loss_options['categories'] = ['log', 'modified_huber']  # Only these loss functions implement predict_proba
            # probabilities assumed uniform if omitited

            penalty_options = OrderedDict()
            penalty_options['distribution'] = 'Categorical'
            penalty_options['categories'] = ['l1', 'l2', 'elasticnet']
            # penalty_options['probabilities'] = [0.05, 0.85, 0.1]  # ?

            alpha_options = OrderedDict()
            alpha_options['distribution'] = 'Uniform'
            alpha_options['lower'] = 5e-5
            alpha_options['upper'] = 5e-4

            l1_ratio_options = OrderedDict()
            l1_ratio_options['distribution'] = 'Uniform'
            l1_ratio_options['lower'] = 0
            l1_ratio_options['upper'] = 1

            fit_intercept_options = OrderedDict()
            fit_intercept_options['distribution'] = 'Categorical'
            fit_intercept_options['categories'] = [True, False]
            fit_intercept_options['probabilities'] = [0.5, 0.5]

            n_iter_options = OrderedDict()
            n_iter_options['distribution'] = 'LogNormal'
            n_iter_options['mean'] = 2.0
            n_iter_options['stdev'] = 1.0
            n_iter_options['categories'] = range(1, 10)

            shuffle_options = OrderedDict()
            shuffle_options['distribution'] = 'Categorical'
            shuffle_options['categories'] = [True, False]

            epsilon_options = OrderedDict()
            epsilon_options['distribution'] = 'LogNormal'
            epsilon_options['mean'] = 5.0
            epsilon_options['stdev'] = 1.0

            learning_rate_options = OrderedDict()
            learning_rate_options['distribution'] = 'Categorical'
            learning_rate_options['categories'] = ['constant', 'optimal', 'invscaling']
            learning_rate_options['probabilities'] = [0.2, 0.3, 0.5]

            eta0_options = OrderedDict()
            eta0_options['distribution'] = 'Uniform'
            eta0_options['lower'] = 0.0
            eta0_options['upper'] = 1.0

            power_t_options = OrderedDict()
            power_t_options['distribution'] = 'Uniform'
            power_t_options['lower'] = 0.0
            power_t_options['upper'] = 1.0

            class_weight_options = OrderedDict()
            class_weight_options['distribution'] = 'Categorical'
            class_weight_options['categories'] = [None, 'auto']
            class_weight_options['probabilities'] = [0.5, 0.5]

            warm_start_options = OrderedDict()
            warm_start_options['distribution'] = 'Categorical'
            warm_start_options['categories'] = [True, False]

            learner_options = {'loss': loss_options,
                               'penalty': penalty_options,
                               'alpha': alpha_options,
                               'l1_ratio': l1_ratio_options,
                               'fit_intercept': fit_intercept_options,
                               'n_iter': n_iter_options,
                               'shuffle': shuffle_options,
                               'epsilon': epsilon_options,
                               'learning_rate': learning_rate_options,
                               'eta0': eta0_options,
                               'power_t': power_t_options,
                               'class_weight': class_weight_options,
                               'warm_start': warm_start_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Categorical parameters
        param_space['loss'] = param.CategoricalParameter('loss',
                                                         ['hinge', 'log', 'modified_huber',
                                                          'squared_hinge', 'perceptron',
                                                          'squared_loss',
                                                          'huber', 'epsilon_insensitive',
                                                          'squared_epsilon_insensitive'],
                                                         default='hinge')
        param_epsilon = param.NumericalParameter('epsilon', prior=param.LogNormal(mean=5.0, stdev=1.0), default=0.1)
        param_space['loss'].categories['huber']['epsilon'] = param_epsilon
        param_space['loss'].categories['epsilon_insensitive']['epsilon'] = param_epsilon
        param_space['loss'].categories['squared_epsilon_insensitive']['epsilon'] = param_epsilon
        param_space['penalty'] = param.CategoricalParameter('penalty', {'l1': 0.05, 'l2': 0.85, 'elasticnet': 0.1}, default='l2')
        param_space['penalty'].categories['elasticnet']['l1_ratio'] = param.NumericalParameter('dual', prior=param.Uniform(lower=0.0, upper=1.0), default=0.15)
        param_space['fit_intercept'] = param.CategoricalParameter('fit_intercept', {True: 0.5, False: 0.5}, default=False)
        param_space['shuffle'] = param.CategoricalParameter('shuffle', {True: 0.5, False: 0.5}, default=False)
        param_space['learning_rate'] = param.CategoricalParameter('learning_rate', {'constant': 0.33333, 'optimal': 0.33333, 'invscaling': 0.33333}, default='optimal')
        param_space['learning_rate'].categories['invscaling']['power_t'] = param.NumericalParameter('power_t', prior=param.Uniform(lower=0.0, upper=1.0), default=0.5)
        param_space['class_weight'] = param.CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)
        param_space['warm_start'] = param.CategoricalParameter('warm_start', {True: 0.5, False: 0.5}, default=False)

        # Numerical parameters
        param_space['alpha'] = param.NumericalParameter('alpha', prior=param.Uniform(5e-5, 5e-4), default=1e-4)
        param_space['n_iter'] = param.NumericalParameter('n_iter', prior=param.LogNormal(mean=np.log(5), stdev=1.0), default=5, discretize=True)
        param_space['eta0'] = param.NumericalParameter('eta0', prior=param.Uniform(lower=0.0, upper=1.0), default=0.0)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Numerical parameters
        param_space['alpha'] = param.NumericalParameter('alpha', prior=param.Uniform(5e-5, 5e-4), default=1e-4)
        param_space['n_iter'] = param.NumericalParameter('n_iter', prior=param.LogNormal(mean=np.log(5), stdev=1.0), default=5, discretize=True)
        param_space['eta0'] = param.NumericalParameter('eta0', prior=param.Uniform(lower=0.0, upper=1.0), default=0.0)

        # Categorical parameters
        loss_cats = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                     'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']

        param_space['loss'] = param.CategoricalParameter('loss', loss_cats, default='hinge')
        loss_branch = param_space['loss'].categories

        param_epsilon = param.NumericalParameter('epsilon', prior=param.LogNormal(mean=5.0, stdev=1.0), default=0.1)
        loss_branch['huber']['epsilon'] = param_epsilon
        loss_branch['epsilon_insensitive']['epsilon'] = param_epsilon
        loss_branch['squared_epsilon_insensitive']['epsilon'] = param_epsilon

        penalty_cats = ['l1', 'l2', 'elasticnet']
        penalty_param = param.CategoricalParameter('penalty', penalty_cats, default='l2')
        penalty_branch = penalty_param.categories
        penalty_branch['elasticnet']['l1_ratio'] = param.NumericalParameter('dual', prior=param.Uniform(lower=0.0, upper=1.0), default=0.15)

        warm_start_param = param.CategoricalParameter('warm_start', {True: 0.5, False: 0.5}, default=False)

        class_weight_cats = [None, 'auto']
        class_weight_param = param.CategoricalParameter('class_weight', class_weight_cats, default=None)
        for class_weight in class_weight_cats:
            class_weight_branch = class_weight_param.categories[class_weight]
            class_weight_branch['warm_start'] = warm_start_param

        learning_rate_cats = ['constant', 'optimal', 'invscaling']
        learning_rate_param = param.CategoricalParameter('learning_rate',  learning_rate_cats, default='optimal')
        for learning_rate in learning_rate_cats:
            learning_rate_branch = learning_rate_param.categories[learning_rate]
            learning_rate_branch['class_weight'] = class_weight_param
        learning_rate_param.categories['invscaling']['power_t'] = param.NumericalParameter('power_t', prior=param.Uniform(lower=0.0, upper=1.0), default=0.5)

        shuffle_cats = [True, False]
        shuffle_param = param.CategoricalParameter('shuffle', shuffle_cats, default=False)
        for shuffle in shuffle_cats:
            shuffle_branch = shuffle_param.categories[shuffle]
            shuffle_branch['learning_rate'] = learning_rate_param

        fit_intercept_cats = [True, False]
        fit_intercept_param = param.CategoricalParameter('fit_intercept', fit_intercept_cats, default=False)
        for fit_intercept in fit_intercept_cats:
            fit_intercept_branch = fit_intercept_param.categories[fit_intercept]
            fit_intercept_branch['shuffle'] = shuffle_param

        for penalty in penalty_cats:
            penalty_branch = penalty_param.categories[penalty]
            penalty_branch['fit_intercept'] = fit_intercept_param

        for loss in loss_cats:
            loss_branch = param_space['loss'].categories[loss]
            loss_branch['penalty'] = penalty_param

        '''
        param_space['fit_intercept'] = param.CategoricalParameter('fit_intercept', {True: 0.5, False: 0.5}, default=False)
        param_space['shuffle'] = param.CategoricalParameter('shuffle', {True: 0.5, False: 0.5}, default=False)
        param_space['learning_rate'] = param.CategoricalParameter('learning_rate', {'constant': 0.33333, 'optimal': 0.33333, 'invscaling': 0.33333}, default='optimal')
        param_space['learning_rate'].categories['invscaling']['power_t'] = param.NumericalParameter('power_t', prior=param.Uniform(lower=0.0, upper=1.0), default=0.5)
        param_space['class_weight'] = param.CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)
        param_space['warm_start'] = param.CategoricalParameter('warm_start', {True: 0.5, False: 0.5}, default=False)
        '''

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['SGDClassifier'][optimizer]

        # Build priors
        loss_prior = Distribution.load(learner_parameters['loss'])
        penalty_prior = Distribution.load(learner_parameters['penalty'])
        alpha_prior = Distribution.load(learner_parameters['alpha'])
        l1_ratio_prior = Distribution.load(learner_parameters['l1_ratio'])
        fit_intercept_prior = Distribution.load(learner_parameters['fit_intercept'], category_type='bool')
        n_iter_prior = Distribution.load(learner_parameters['n_iter'], category_type='int')
        shuffle_prior = Distribution.load(learner_parameters['shuffle'], category_type='bool')
        epsilon_prior = Distribution.load(learner_parameters['epsilon'])
        learning_rate_prior = Distribution.load(learner_parameters['learning_rate'])
        eta0_prior = Distribution.load(learner_parameters['eta0'])
        power_t_prior = Distribution.load(learner_parameters['power_t'])
        class_weight_prior = Distribution.load(learner_parameters['class_weight'], parse_none=True)
        warm_start_prior = Distribution.load(learner_parameters['warm_start'], category_type='bool')

        # Build parameters
        loss_param = Parameter('loss', prior=loss_prior, default='hinge')
        penalty_param = Parameter('penalty', prior=penalty_prior, default='l2')
        alpha_param = Parameter('alpha', prior=alpha_prior, default=0.0001)
        l1_ratio_param = Parameter('l1_ratio', prior=l1_ratio_prior, default=0.15,
                                   sample_if=lambda params: params['penalty'] == 'elasticnet')
        fit_intercept_param = Parameter('fit_intercept', prior=fit_intercept_prior, default=True)
        n_iter_param = Parameter('n_iter', prior=n_iter_prior, default=5)
        shuffle_param = Parameter('shuffle', prior=shuffle_prior, default=False)
        epsilon_param = Parameter('epsilon', prior=epsilon_prior, default=0.1,
                                  sample_if=lambda params: params['loss'] in ('huber', 'epsilon_insensitive', 'square_epsilon_insensitive'))
        learning_rate_param = Parameter('learning_rate', prior=learning_rate_prior, default='optimal')
        eta0_param = Parameter('eta0', prior=eta0_prior, default=0.0)
        power_t_param = Parameter('power_t', prior=power_t_prior, default=0.5,
                                  sample_if=lambda params: params['learning_rate'] == 'invscaling')
        class_weight_param = Parameter('class_weight', prior=class_weight_prior, default=None)
        warm_start_param = Parameter('warm_start', prior=warm_start_prior, default=False)

        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['loss'] = loss_param
        param_space['penalty'] = penalty_param
        param_space['alpha'] = alpha_param
        param_space['l1_ratio'] = l1_ratio_param
        param_space['fit_intercept'] = fit_intercept_param
        param_space['n_iter'] = n_iter_param
        param_space['shuffle'] = shuffle_param
        param_space['epsilon'] = epsilon_param
        param_space['learning_rate'] = learning_rate_param
        param_space['eta0'] = eta0_param
        param_space['power_t'] = power_t_param
        param_space['class_weight'] = class_weight_param
        param_space['warm_start'] = warm_start_param
        param_space['random_state'] = random_state_param
        return param_space

    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        if self.classifier.loss in ('log', 'modified_huber'):
            prediction = super(SGDClassifier, self).predict(data)
        else:
            self.predict_data = data
            sgd = self.classifier
            prediction = sgd.predict(data)
            prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction


class PassiveAggressiveClassifier(BaseClassifier):
    """Passive Aggressive classifier implementation."""

    def create_classifier(self, **parameters):
        """Creates the passive agressive classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Passive Agressive classifier.
        """
        classifier = SKPassiveAggressiveClassifier(**parameters)
        return classifier

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['PassiveAggressiveClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            C_options = OrderedDict()
            C_options['distribution'] = 'Uniform'
            C_options['lower'] = 0.0
            C_options['upper'] = 5.0

            fit_intercept_options = OrderedDict()
            fit_intercept_options['distribution'] = 'Categorical'
            fit_intercept_options['categories'] = [True, False]
            fit_intercept_options['probabilities'] = [0.5, 0.5]

            n_iter_options = OrderedDict()
            n_iter_options['distribution'] = 'LogNormal'
            n_iter_options['mean'] = 2.0
            n_iter_options['stdev'] = 1.0
            n_iter_options['categories'] = range(1, 10)

            shuffle_options = OrderedDict()
            shuffle_options['distribution'] = 'Categorical'
            shuffle_options['categories'] = [True, False]
            shuffle_options['probabilities'] = [0.5, 0.5]

            loss_options = OrderedDict()
            loss_options['distribution'] = 'Categorical'
            loss_options['categories'] = ['hinge', 'squared_hinge']
            loss_options['probabilities'] = [0.5, 0.5]

            warm_start_options = OrderedDict()
            warm_start_options['distribution'] = 'Categorical'
            warm_start_options['categories'] = [True, False]
            warm_start_options['probabilities'] = [0.5, 0.5]

            learner_options = {'C': C_options,
                               'fit_intercept': fit_intercept_options,
                               'n_iter': n_iter_options,
                               'shuffle': shuffle_options,
                               'loss': loss_options,
                               'warm_start': warm_start_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Categorical parameters
        param_space['loss'] = param.CategoricalParameter('loss', ['hinge', 'squared_hinge'], default='hinge')
        param_space['fit_intercept'] = param.CategoricalParameter('fit_intercept', {True: 0.5, False: 0.5}, default=True)
        param_space['shuffle'] = param.CategoricalParameter('shuffle', {True: 0.5, False: 0.5}, default=False)
        param_space['warm_start'] = param.CategoricalParameter('warm_start', {True: 0.5, False: 0.5}, default=False)

        # Numerical parameters
        param_space['C'] = param.NumericalParameter('C', prior=param.LogUniform(-10.0, 10.0), default=1.0)
        param_space['n_iter'] = param.NumericalParameter('n_iter', prior=param.LogNormal(mean=0.0, stdev=1.0),
                                                         default=1, discretize=True,
                                                         valid_if="value > 0", when_invalid='resample')

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Numerical parameters
        param_space['C'] = param.NumericalParameter('C', prior=param.LogUniform(-10.0, 10.0), default=1.0)
        param_space['n_iter'] = param.NumericalParameter('n_iter', prior=param.LogNormal(mean=0.0, stdev=1.0),
                                                         default=1, discretize=True,
                                                         valid_if="value > 0", when_invalid='resample')

        # Categorical parameters

        warm_start_param = param.CategoricalParameter('warm_start', {True: 0.5, False: 0.5}, default=False)

        shuffle_cats = [True, False]
        shuffle_param = param.CategoricalParameter('shuffle', shuffle_cats, default=False)
        for shuffle in shuffle_cats:
            shuffle_branch = shuffle_param.categories[shuffle]
            shuffle_branch['warm_start'] = warm_start_param

        fit_intercept_cats = [True, False]
        fit_intercept_param = param.CategoricalParameter('fit_intercept', fit_intercept_cats, default=True)
        for fit_intercept in fit_intercept_cats:
            fit_intercept_branch = fit_intercept_param.categories[fit_intercept]
            fit_intercept_branch['shuffle'] = shuffle_param

        loss_cats = ['hinge', 'squared_hinge']
        loss_param = param.CategoricalParameter('loss', loss_cats, default='hinge')
        for loss in loss_cats:
            loss_branch = loss_param.categories[loss]
            loss_branch['fit_intercept'] = fit_intercept_param

        param_space['loss'] = loss_param

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['PassiveAggressiveClassifier'][optimizer]

        # Build priors
        C_prior = Distribution.load(learner_parameters['C'])
        fit_intercept_prior = Distribution.load(learner_parameters['fit_intercept'], category_type='bool')
        n_iter_prior = Distribution.load(learner_parameters['n_iter'], category_type='int')
        shuffle_prior = Distribution.load(learner_parameters['shuffle'], category_type='bool')
        loss_prior = Distribution.load(learner_parameters['loss'])
        warm_start_prior = Distribution.load(learner_parameters['warm_start'], category_type='bool')

        # Build parameters
        C_param = Parameter('C', prior=C_prior, default=1.0)
        fit_intercept_param = Parameter('fit_intercept', prior=fit_intercept_prior, default=True)
        n_iter_param = Parameter('n_iter', prior=n_iter_prior, default=5)
        shuffle_param = Parameter('shuffle', prior=shuffle_prior, default=False)
        loss_param = Parameter('loss', prior=loss_prior, default='hinge')
        warm_start_param = Parameter('warm_start', prior=warm_start_prior, default=False)

        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['C'] = C_param
        param_space['fit_intercept'] = fit_intercept_param
        param_space['n_iter'] = n_iter_param
        param_space['shuffle'] = shuffle_param
        param_space['loss'] = loss_param
        param_space['warm_start'] = warm_start_param
        param_space['random_state'] = random_state_param
        return param_space

    #@log_step("Predicting with Passive-aggressive classifier")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        pac = self.classifier
        prediction = pac.predict(data)
        prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction


class RidgeClassifier(BaseClassifier):
    """Ridge classifier implementation."""

    def create_classifier(self, **parameters):
        """Creates the Ridge classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Ridge classifier.
        """
        classifier = SKRidgeClassifier(**parameters)
        return classifier

    #@log_step("Predicting with Ridge classifier")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        ridge = self.classifier
        prediction = ridge.predict(data)
        prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            alpha_options = OrderedDict()
            alpha_options['distribution'] = 'LogNormal'
            alpha_options['mean'] = 1.0
            alpha_options['stdev'] = 0.5

            # class_weight_options = OrderedDict()
            # class_weight_options['distribution'] = 'Categorical'
            # class_weight_options['categories'] = [None]  # TODO Provide interface to assign weights
            # class_weight_options['probabilities'] = [1]

            fit_intercept_options = OrderedDict()
            fit_intercept_options['distribution'] = 'Categorical'
            fit_intercept_options['categories'] = [True, False]
            fit_intercept_options['probabilities'] = [0.5, 0.5]

            max_iter_options = OrderedDict()
            max_iter_options['distribution'] = 'Categorical'
            max_iter_options['categories'] = range(1, 10)

            normalize_options = OrderedDict()
            normalize_options['distribution'] = 'Categorical'
            normalize_options['categories'] = [True, False]
            normalize_options['probabilities'] = [0.5, 0.5]

            solver_options = OrderedDict()
            solver_options['distribution'] = 'Categorical'
            solver_options['categories'] = ['svd', 'dense_cholesky', 'lsqr', 'sparse_cg']

            tol_options = OrderedDict()
            tol_options['distribution'] = 'LogNormal'
            tol_options['mean'] = 0.001
            tol_options['stdev'] = 0.5

            learner_options = {'alpha': alpha_options,
                               # 'class_weight': class_weight_options,
                               'fit_intercept': fit_intercept_options,
                               'max_iter': max_iter_options,
                               'normalize': normalize_options,
                               'solver': solver_options,
                               'tolerance': tol_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['RidgeClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Categorical parameters
        param_space['fit_intercept'] = param.CategoricalParameter('fit_intercept', [True, False], default=True)
        param_space['normalize'] = param.CategoricalParameter('fit_intercept', [True, False], default=True)
        param_space['solver'] = param.CategoricalParameter('solver', ['svd', 'dense_cholesky', 'lsqr', 'sparse_cg'], default='auto')

        # Numerical parameters
        param_space['alpha'] = param.NumericalParameter('alpha', prior=param.LogNormal(mean=0.0, stdev=0.5), default=1.0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=np.log(0.001), stdev=0.5), default=0.001)
        param_space['max_iter'] = param.NumericalParameter('max_iter', prior=param.LogNormal(mean=np.log(5), stdev=1.0), default=None, discretize=True)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Categorical parameters
        '''
        param_space['fit_intercept'] = param.CategoricalParameter('fit_intercept', [True, False], default=True)
        param_space['normalize'] = param.CategoricalParameter('fit_intercept', [True, False], default=True)
        param_space['solver'] = param.CategoricalParameter('solver', ['svd', 'dense_cholesky', 'lsqr', 'sparse_cg'], default='auto')
        '''

        # Numerical parameters
        param_space['alpha'] = param.NumericalParameter('alpha', prior=param.LogNormal(mean=0.0, stdev=0.5), default=1.0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=np.log(0.001), stdev=0.5), default=0.001)
        param_space['max_iter'] = param.NumericalParameter('max_iter', prior=param.LogNormal(mean=np.log(5), stdev=1.0), default=None, discretize=True)

        normalize_cats = [True, False]
        normalize_param = param.CategoricalParameter('normalize', normalize_cats, default=True)

        fit_intercept_cats = [True, False]
        fit_intercept_param = param.CategoricalParameter('fit_intercept', fit_intercept_cats, default=True)
        for fit_intercept in fit_intercept_cats:
            fit_intercept_branch = fit_intercept_param.categories[fit_intercept]
            fit_intercept_branch['normalize'] = normalize_param

        solver_cats = ['svd', 'dense_cholesky', 'lsqr', 'sparse_cg']
        solver_param = param.CategoricalParameter('solver', solver_cats, default='auto')
        for solver in solver_cats:
            solver_branch = solver_param.categories[solver]
            solver_branch['fit_intercept'] = fit_intercept_param

        param_space['solver'] = solver_param

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['RidgeClassifier'][optimizer]

        # Build priors
        alpha_prior = Distribution.load(learner_parameters['alpha'])
        # class_weight_prior = Distribution.load(learner_parameters['class_weight'], parse_none=True)
        fit_intercept_prior = Distribution.load(learner_parameters['fit_intercept'], category_type='bool')
        max_iter_prior = Distribution.load(learner_parameters['max_iter'], category_type='int')
        normalize_prior = Distribution.load(learner_parameters['normalize'], category_type='bool')
        solver_prior = Distribution.load(learner_parameters['solver'])
        tol_prior = Distribution.load(learner_parameters['tolerance'])

        # Build parameters
        alpha_param = Parameter('alpha', prior=alpha_prior, default=1.0)
        # class_weight_param = Parameter('class_weight', prior=class_weight_prior, default=None)
        fit_intercept_param = Parameter('fit_intercept', prior=fit_intercept_prior, default=True)
        max_iter_param = Parameter('max_iter', prior=max_iter_prior, default=None)
        normalize_param = Parameter('normalize', prior=normalize_prior, default=False)
        solver_param = Parameter('solver', prior=solver_prior, default='auto')
        tol_param = Parameter('tol', prior=tol_prior, default=0.001)

        # Create parameter space
        param_space = ParameterSpace()
        param_space['alpha'] = alpha_param
        # param_space['class_weight'] = class_weight_param
        param_space['fit_intercept'] = fit_intercept_param
        param_space['max_iter'] = max_iter_param
        param_space['normalize'] = normalize_param
        param_space['solver'] = solver_param
        param_space['tol'] = tol_param
        return param_space

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['alpha'] = ChoiceParameter([1.0])
        # param_space['weights'] = ?
        param_space['fit_intercept'] = BooleanParameter()
        param_space['max_iter'] = ChoiceParameter([5])
        param_space['tol'] = ChoiceParameter([0.01])
        # param_space['class_weight'] = ?
        param_space['solver'] = ChoiceParameter(['auto', 'svd', 'dense_cholesky', 'lsqr', 'sparse_cg'])

        return param_space


class RidgeCVClassifier(BaseClassifier):
    """Ridge classifier implementation (with built-in cross-validation)."""

    def create_classifier(self, **parameters):
        """Creates the Ridge classifier class (with built-in cross-validation), with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Ridge classifier (with built-in cross-validation).
        """
        classifier = RidgeClassifierCV(**parameters)
        return classifier

    #@log_step("Predicting with Ridge classifier (built-in CV)")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        ridgecv = self.classifier
        prediction = ridgecv.predict(data)
        prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        # param_space['alphas'] = ? (one per fold?)
        # param_space['class_weight'] = ?
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        # param_space['scoring'] = ?

        return param_space


class GaussianNBClassifier(BaseClassifier):
    """Gaussian Naive Bayes classifier implementation."""
    def create_classifier(self, **parameters):
        """Creates the linear Gaussian Naive Bayes classifier class, with the options given.

        :param parameters: parameters to pass to the constructor.
        :returns: Gaussian Naive Bayes classifier.
        """
        classifier = GaussianNB()
        #Log.write("{classifier}: {params}".format(classifier=self.__class__.__name__, params=parameters))
        return classifier

    #@log_step("Predicting with Gaussian Naive Bayes classifier")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        gaussian_naive_bayes = self.classifier
        prediction = gaussian_naive_bayes.predict_proba(data)
        return prediction

    @classmethod
    def create_default_params(self):
        return ParameterSpace()

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            learner_options = {'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['GaussianNBClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        param_space = ParameterSpace()
        return param_space


class MultinomialNBClassifier(BaseClassifier):
    """Naive Bayes classifier for multinomial models."""

    def create_classifier(self, **parameters):
        """Creates the Multinomial naive Bayes classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Multinomial naive Bayes classifier.
        """
        classifier = MultinomialNB(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        #param_space['alpha'] = ChoiceParameter([1.0])
        #param_space['fit_prior'] = BooleanParameter()
        # param_space['class_prior'] = ?

        return param_space


class BernoulliNBClassifier(BaseClassifier):
    """Naive Bayes classifier for multivariate Bernoulli models."""

    def create_classifier(self, **parameters):
        """Creates the Bernoulli naive Bayes classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Bernoulli naive Bayes classifier.
        """
        classifier = BernoulliNB(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['alpha'] = ChoiceParameter([1.0])
        param_space['binarize'] = ChoiceParameter([0.0])
        param_space['fit_prior'] = BooleanParameter()
        # param_space['class_prior'] = ?

        return param_space


class KNNClassifier(BaseClassifier):
    """K-Nearest-Neighbors classifier implementation."""
    def get_learner_parameters(self, **parameters):
        return (), parameters

    def create_classifier(self, **parameters):
        """Creates the k-Nearest-Neighbors classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: k-Nearest-Neighbors classifier.
        """
        required_params, optional_params = self.get_learner_parameters(**parameters)
        classifier = KNeighborsClassifier(*required_params, **optional_params)
        #Log.write("{classifier}: {params}".format(classifier=self.__class__.__name__, params=parameters))
        return classifier

    #@log_step("Predicting with Knn classifier")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        knn = self.classifier
        #prediction = knn.predict_proba(dataset.data)
        prediction = knn.predict(data)
        prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()
        #"""
        param_space['n_neighbors'] = ChoiceParameter([3, 4, 5, 6, 7, 8, 9, 10])
        param_space['weights'] = ChoiceParameter(['uniform', 'distance'])
        param_space['metric'] = ChoiceParameter(['euclidean', 'manhattan', 'chebyshev'])
        param_space['algorithm'] = ChoiceParameter(['brute', 'ball_tree', 'kd_tree'])
        """
        param_space['n_neighbors'] = ChoiceParameter([2, 3, 4, 5, 6])
        param_space['weights'] = ChoiceParameter(['uniform', 'distance'])
        param_space['metric'] = ChoiceParameter(['euclidean', 'manhattan', 'chebyshev'])
        param_space['algorithm'] = ChoiceParameter(['brute'])
        """
        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            n_neighbors_options = OrderedDict()
            n_neighbors_options['distribution'] = 'LogNormal'
            n_neighbors_options['mean'] = 1.0
            n_neighbors_options['stdev'] = 0.5
            n_neighbors_options['categories'] = range(2, 20)

            weights_options = OrderedDict()
            weights_options['distribution'] = 'Categorical'
            weights_options['categories'] = ['uniform', 'distance']
            weights_options['probabilities'] = [0.5, 0.5]

            algorithm_options = OrderedDict()
            algorithm_options['distribution'] = 'Categorical'
            algorithm_options['categories'] = ['ball_tree', 'kd_tree', 'brute']

            leaf_size_options = OrderedDict()
            leaf_size_options['distribution'] = 'Categorical'  # ?
            leaf_size_options['categories'] = range(15, 45)

            metric_options = OrderedDict()
            metric_options['distribution'] = 'Categorical'
            metric_options['categories'] = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean']  # 'mahalanobis'?

            p_options = OrderedDict()
            p_options['distribution'] = 'Categorical'
            p_options['categories'] = range(1, 20)

            w_options = OrderedDict()
            w_options['distribution'] = 'Uniform'
            w_options['lower'] = 0.0
            w_options['upper'] = 1.0

            learner_options = {'n_neighbors': n_neighbors_options,
                               'weights': weights_options,
                               'algorithm': algorithm_options,
                               'leaf_size': leaf_size_options,
                               'p': p_options,
                               'w': w_options,
                               'metric': metric_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['KNNClassifier'][optimizer]

        # Build priors
        n_neighbors_prior = Distribution.load(learner_parameters['n_neighbors'], category_type='int')
        weights_prior = Distribution.load(learner_parameters['weights'])
        algorithm_prior = Distribution.load(learner_parameters['algorithm'])
        leaf_size_prior = Distribution.load(learner_parameters['leaf_size'], category_type='int')
        metric_prior = Distribution.load(learner_parameters['metric'])
        p_prior = Distribution.load(learner_parameters['p'], category_type='int')
        w_prior = Distribution.load(learner_parameters['w'])

        # Build parameters
        n_neighbors_param = Parameter('n_neighbors', prior=n_neighbors_prior, default=5)
        weights_param = Parameter('weights', prior=weights_prior, default='uniform')
        algorithm_param = Parameter('algorithm', prior=algorithm_prior, default='auto')
        leaf_size_param = Parameter('leaf_size', prior=leaf_size_prior, default=30,
                                    sample_if=lambda params: params['algorithm'] in ('ball_tree', 'kd_tree'))
        metric_param = Parameter('metric', prior=metric_prior, default='minkowski',
                                 valid_if=lambda value, params: (params['algorithm'] != 'kd_tree') or (value not in ('seuclidean', 'wminkowski')),
                                 action_if_invalid='resample')
        p_param = Parameter('p', prior=p_prior, default=2,
                            sample_if=lambda params: params['metric'] in ('minkowski', 'wminkowski'))
        w_param = Parameter('w', prior=w_prior, default=1,
                            sample_if=lambda params: params['metric'] == 'wminkowski')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['n_neighbors'] = n_neighbors_param
        param_space['weights'] = weights_param
        param_space['algorithm'] = algorithm_param
        param_space['leaf_size'] = leaf_size_param
        param_space['metric'] = metric_param
        param_space['p'] = p_param
        param_space['w'] = w_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['KNNClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg_2(self):
        # WMINKOWSKI IS PROBLEMATIC
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Categorical parameters
        param_space['weights'] = param.CategoricalParameter('weights', ['uniform', 'distance'], default='uniform')
        param_space['algorithm'] = param.CategoricalParameter('algorithm', ['ball_tree', 'kd_tree', 'brute'], default='auto')
        #param_ball_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'seuclidean', 'minkowski', 'wminkowski'], default='minkowski')
        param_ball_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_ball_tree_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_ball_tree_metric.categories['wminkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_ball_tree_metric.categories['wminkowski']['w'] = param.NumericalParameter('w', prior=param.Uniform(0.0, 1.0), default=0.5)
        param_kd_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_kd_tree_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_brute_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'seuclidean', 'minkowski', 'wminkowski'], default='minkowski')
        param_brute_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_brute_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_brute_metric.categories['wminkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_brute_metric.categories['wminkowski']['w'] = param.NumericalParameter('w', prior=param.Uniform(0.0, 1.0), default=0.5)

        param_space['algorithm'].categories['ball_tree']['metric'] = param_ball_tree_metric
        param_space['algorithm'].categories['ball_tree']['leaf_size'] = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        param_space['algorithm'].categories['kd_tree']['metric'] = param_kd_tree_metric
        param_space['algorithm'].categories['kd_tree']['leaf_size'] = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        param_space['algorithm'].categories['brute']['metric'] = param_brute_metric

        # Numerical parameters
        param_space['n_neighbors'] = param.NumericalParameter('n_neighbors', prior=param.LogNormal(mean=np.log(5), stdev=1.0), default=5, discretize=True)
        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        # WMINKOWSKI IS PROBLEMATIC
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Categorical parameters
        algorithm_cats = ['ball_tree', 'kd_tree', 'brute']
        param_space['algorithm'] = param.CategoricalParameter('algorithm', algorithm_cats, default='auto')
        for algorithm in algorithm_cats:
            weights_cats = ['uniform', 'distance']
            param_space['algorithm'].categories[algorithm]['weights'] = param.CategoricalParameter('weights', weights_cats, default='uniform')
        kd_ball_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        kd_ball_tree_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        brute_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        brute_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        for weight in param_space['algorithm'].categories['ball_tree']['weights'].categories.values():
            weight['metric'] = kd_ball_tree_metric
            weight['leaf_size'] = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        for weight in param_space['algorithm'].categories['kd_tree']['weights'].categories.values():
            weight['metric'] = kd_ball_tree_metric
            weight['leaf_size'] = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        for weight in param_space['algorithm'].categories['brute']['weights'].categories.values():
            weight['metric'] = brute_metric

        # Numerical parameters
        param_space['n_neighbors'] = param.NumericalParameter('n_neighbors', prior=param.LogNormal(mean=np.log(5), stdev=1.0), default=5, discretize=True)
        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class RadiusNeighborsClassifier(BaseClassifier):
    """Classifier based on neighbors voting within a given radius."""

    def create_classifier(self, **parameters):
        """Creates the radius neighbors classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Radius neighbors classifier.
        """
        classifier = SKRadiusNeighborsClassifier(**parameters)
        return classifier

    #@log_step("Predicting with radius neighbors classifier")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        classifier = self.classifier
        prediction = classifier.predict(data)
        prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['radius'] = ChoiceParameter([1.0])
        param_space['weights'] = ChoiceParameter(['uniform', 'distance'])
        param_space['algorithm'] = ChoiceParameter(['brute', 'ball_tree', 'kd_tree'])  # 'auto'??
        param_space['metric'] = ChoiceParameter(['euclidean', 'manhattan', 'chebyshev',
                                                 'minkowski', 'wminkowski', 'seuclidean',
                                                 'mahalanobis'])
        param_space['p'] = ChoiceParameter([2])
        #param_space['outlier_label'] = ?
        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            radius_options = OrderedDict()
            radius_options['distribution'] = 'LogUniform'
            radius_options['lower'] = -5.0
            radius_options['upper'] = 5.0

            weights_options = OrderedDict()
            weights_options['distribution'] = 'Categorical'
            weights_options['categories'] = ['uniform', 'distance']
            weights_options['probabilities'] = [0.5, 0.5]

            algorithm_options = OrderedDict()
            algorithm_options['distribution'] = 'Categorical'
            algorithm_options['categories'] = ['ball_tree', 'kd_tree', 'brute']

            leaf_size_options = OrderedDict()
            leaf_size_options['distribution'] = 'Categorical'  # ?
            leaf_size_options['categories'] = range(15, 45)

            metric_options = OrderedDict()
            metric_options['distribution'] = 'Categorical'
            metric_options['categories'] = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean']  # 'mahalanobis'?

            p_options = OrderedDict()
            p_options['distribution'] = 'Categorical'
            p_options['categories'] = range(1, 20)

            w_options = OrderedDict()
            w_options['distribution'] = 'Uniform'
            w_options['lower'] = 0.0
            w_options['upper'] = 1.0

            learner_options = {'radius': radius_options,
                               'weights': weights_options,
                               'algorithm': algorithm_options,
                               'leaf_size': leaf_size_options,
                               'p': p_options,
                               'w': w_options,
                               'metric': metric_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['RadiusNeighborsClassifier'][optimizer]

        # Build priors
        radius_prior = Distribution.load(learner_parameters['radius'])
        weights_prior = Distribution.load(learner_parameters['weights'])
        algorithm_prior = Distribution.load(learner_parameters['algorithm'])
        leaf_size_prior = Distribution.load(learner_parameters['leaf_size'], category_type='int')
        metric_prior = Distribution.load(learner_parameters['metric'])
        p_prior = Distribution.load(learner_parameters['p'], category_type='int')
        w_prior = Distribution.load(learner_parameters['w'])

        # Build parameters
        radius_param = Parameter('n_neighbors', prior=radius_prior, default=1.0)
        weights_param = Parameter('weights', prior=weights_prior, default='uniform')
        algorithm_param = Parameter('algorithm', prior=algorithm_prior, default='auto')
        leaf_size_param = Parameter('leaf_size', prior=leaf_size_prior, default=30,
                                    sample_if=lambda params: params['algorithm'] in ('ball_tree', 'kd_tree'))
        metric_param = Parameter('metric', prior=metric_prior, default='minkowski',
                                 valid_if=lambda value, params: (params['algorithm'] != 'kd_tree') or (value not in ('seuclidean', 'wminkowski')),
                                 action_if_invalid='resample')
        p_param = Parameter('p', prior=p_prior, default=2,
                            sample_if=lambda params: params['metric'] in ('minkowski', 'wminkowski'))
        w_param = Parameter('w', prior=w_prior, default=1,
                            sample_if=lambda params: params['metric'] == 'wminkowski')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['radius'] = radius_param
        param_space['weights'] = weights_param
        param_space['algorithm'] = algorithm_param
        param_space['leaf_size'] = leaf_size_param
        param_space['metric'] = metric_param
        param_space['p'] = p_param
        param_space['w'] = w_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['RadiusNeighborsClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        # WMINKOWSKI IS PROBLEMATIC
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Categorical parameters
        param_space['weights'] = param.CategoricalParameter('weights', ['uniform', 'distance'], default='uniform')
        param_space['algorithm'] = param.CategoricalParameter('algorithm', ['ball_tree', 'kd_tree', 'brute'], default='auto')
        #param_ball_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'seuclidean', 'minkowski', 'wminkowski'], default='minkowski')
        param_ball_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_ball_tree_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_ball_tree_metric.categories['wminkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_ball_tree_metric.categories['wminkowski']['w'] = param.NumericalParameter('w', prior=param.Uniform(0.0, 1.0), default=0.5)
        param_kd_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_kd_tree_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_brute_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'seuclidean', 'minkowski', 'wminkowski'], default='minkowski')
        param_brute_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_brute_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_brute_metric.categories['wminkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        #param_brute_metric.categories['wminkowski']['w'] = param.NumericalParameter('w', prior=param.Uniform(0.0, 1.0), default=0.5)

        param_space['algorithm'].categories['ball_tree']['metric'] = param_ball_tree_metric
        param_space['algorithm'].categories['ball_tree']['leaf_size'] = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        param_space['algorithm'].categories['kd_tree']['metric'] = param_kd_tree_metric
        param_space['algorithm'].categories['kd_tree']['leaf_size'] = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        param_space['algorithm'].categories['brute']['metric'] = param_brute_metric

        # Numerical parameters
        param_space['radius'] = param.NumericalParameter('radius', prior=param.LogUniform(lower=-10.0, upper=10.0), default=1.0)
        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        # WMINKOWSKI IS PROBLEMATIC
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Categorical parameters
        '''
        param_space['weights'] = param.CategoricalParameter('weights', ['uniform', 'distance'], default='uniform')
        param_space['algorithm'] = param.CategoricalParameter('algorithm', ['ball_tree', 'kd_tree', 'brute'], default='auto')
        param_ball_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_ball_tree_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        param_kd_tree_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_kd_tree_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)
        param_brute_metric = param.CategoricalParameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], default='minkowski')
        param_brute_metric.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)

        param_space['algorithm'].categories['ball_tree']['metric'] = param_ball_tree_metric
        param_space['algorithm'].categories['ball_tree']['leaf_size'] = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        param_space['algorithm'].categories['kd_tree']['metric'] = param_kd_tree_metric
        param_space['algorithm'].categories['kd_tree']['leaf_size'] = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        param_space['algorithm'].categories['brute']['metric'] = param_brute_metric
        '''

        weights_cats = ['uniform', 'distance']
        weights_param = param.CategoricalParameter('weights', weights_cats, default='uniform')

        metric_cats = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        metric_param = param.CategoricalParameter('metric', metric_cats, default='minkowski')
        metric_param.categories['minkowski']['p'] = param.NumericalParameter('p', prior=param.Uniform(1, 20), default=2, discretize=True)

        for metric in metric_cats:
            metric_branch = metric_param.categories[metric]
            metric_branch['weights'] = weights_param

        algorithm_cats = ['ball_tree', 'kd_tree', 'brute']
        algorithm_param = param.CategoricalParameter('algorithm', algorithm_cats, default='auto')
        leaf_size_param = param.NumericalParameter('leaf_size', prior=param.Uniform(lower=15, upper=45), default=30, discretize=True)
        algorithm_param.categories['ball_tree']['leaf_size'] = leaf_size_param
        algorithm_param.categories['kd_tree']['leaf_size'] = leaf_size_param
        for algorithm in algorithm_cats:
            algorithm_branch = algorithm_param.categories[algorithm]
            algorithm_branch['metric'] = metric_param

        param_space['algorithm'] = algorithm_param

        # Numerical parameters
        param_space['radius'] = param.NumericalParameter('radius', prior=param.LogUniform(lower=-10.0, upper=10.0), default=1.0)
        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class NearestCentroidClassifier(BaseClassifier):
    """Nearest centroid classifier."""

    def create_classifier(self, **parameters):
        """Creates the nearest centroid class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Nearest centroid classifier.
        """
        classifier = NearestCentroid(**parameters)
        return classifier

    #@log_step("Predicting with radius neighbors classifier")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        classifier = self.classifier
        prediction = classifier.predict(data)
        prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            metric_options = OrderedDict()
            metric_options['distribution'] = 'Categorical'
            metric_options['categories'] = ['euclidean', 'l2', 'l1', 'manhattan',
                                            'cityblock',  # not same as manhattan???
                                            'braycurtis', 'canberra', 'chebyshev',
                                            'correlation', 'cosine', 'dice', 'hamming',
                                            'jaccard', 'kulsinski', 'mahalanobis',
                                            'matching', 'minkowski', 'rogerstanimoto',
                                            'russellrao', 'seuclidean', 'sokalmichener',
                                            'sokalsneath', 'sqeuclidean', 'yule']

            shrink_threshold_options = OrderedDict()
            shrink_threshold_options['distribution'] = 'LogUniform'
            shrink_threshold_options['lower'] = -10.0
            shrink_threshold_options['upper'] = 10.0

            learner_options = {'metric': metric_options,
                               'shrink_threshold': shrink_threshold_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['metric'] = ChoiceParameter(['euclidean', 'l2', 'l1', 'manhattan',
                                                 'cityblock',  # not same as manhattan???
                                                 'braycurtis', 'canberra', 'chebyshev',
                                                 'correlation', 'cosine', 'dice', 'hamming',
                                                 'jaccard', 'kulsinski', 'mahalanobis',
                                                 'matching', 'minkowski', 'rogerstanimoto',
                                                 'russellrao', 'seuclidean', 'sokalmichener',
                                                 'sokalsneath', 'sqeuclidean', 'yule'])
        param_space['shrink_threshold'] = ChoiceParameter([1.0])  # default is None

        return param_space

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['NearestCentroidClassifier'][optimizer]

        # Build priors
        metric_prior = Distribution.load(learner_parameters['metric'])
        shrink_threshold_prior = Distribution.load(learner_parameters['shrink_threshold'])

        # Build parameters
        metric_param = Parameter('metric', prior=metric_prior, default='euclidean')
        shrink_threshold_param = Parameter('shrink_threshold', prior=shrink_threshold_prior, default=None)
        # Create parameter space
        param_space = ParameterSpace()
        param_space['metric'] = metric_param
        param_space['shrink_threshold'] = shrink_threshold_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['NearestCentroidClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['metric'] = param.CategoricalParameter('metric', ['euclidean', 'l1', 'l2', 'manhattan', 'cityblock', 'braycurtis',
                                                                      'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming',
                                                                      'jaccard', 'kulsinski', 'mahalanobis',
                                                                      'matching', 'minkowski', 'rogerstanimoto',
                                                                      'russellrao', 'seuclidean', 'sokalmichener',
                                                                      'sokalsneath', 'sqeuclidean', 'yule'], default='euclidean')
        param_space['shrink_threshold'] = param.NumericalParameter('shrink_threshold', prior=param.LogUniform(lower=-10.0, upper=10.0), default=None)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['metric'] = param.CategoricalParameter('metric', ['euclidean', 'l1', 'l2', 'manhattan', 'cityblock', 'braycurtis',
                                                                      'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming',
                                                                      'jaccard', 'kulsinski', 'mahalanobis',
                                                                      'matching', 'minkowski', 'rogerstanimoto',
                                                                      'russellrao', 'seuclidean', 'sokalmichener',
                                                                      'sokalsneath', 'sqeuclidean', 'yule'], default='euclidean')
        param_space['shrink_threshold'] = param.NumericalParameter('shrink_threshold', prior=param.LogUniform(lower=-10.0, upper=10.0), default=None)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class DecisionTreeClassifier(BaseClassifier):
    """Decision tree classifier."""

    def create_classifier(self, **parameters):
        """Creates the decision tree classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Decision tree classifier.
        """
        if 'max_features_use_preset' in parameters:
            del parameters['max_features_use_preset']
            if 'max_features_preset' in parameters:
                parameters['max_features'] = parameters['max_features_preset']
                del parameters['max_features_preset']
            else:
                parameters['max_features'] = parameters['max_features_sample']
                del parameters['max_features_sample']

        classifier = SKDecisionTreeClassifier(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['criterion'] = ChoiceParameter(['gini', 'entropy'])
        #param_space['max_features'] = ?
        param_space['max_depth'] = ChoiceParameter([3])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        #param_space['random_state'] = ?

        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            criterion_options = OrderedDict()
            criterion_options['distribution'] = 'Categorical'
            criterion_options['categories'] = ['gini', 'entropy']
            criterion_options['probabilities'] = [0.5, 0.5]

            max_features_use_preset_options = OrderedDict()
            max_features_use_preset_options['distribution'] = 'Categorical'
            max_features_use_preset_options['categories'] = [True, False]
            max_features_use_preset_options['probabilities'] = [0.75, 0.25]

            max_features_preset_options = OrderedDict()
            max_features_preset_options['distribution'] = 'Categorical'
            max_features_preset_options['categories'] = ['sqrt', 'log2', None]

            max_features_sample_options = OrderedDict()
            max_features_sample_options['distribution'] = 'Uniform'
            max_features_sample_options['lower'] = 0.0
            max_features_sample_options['upper'] = 1.0

            max_depth_options = OrderedDict()
            max_depth_options['distribution'] = 'LogUniform'
            max_depth_options['lower'] = 0.0
            max_depth_options['upper'] = 5.0
            # TODO add discretization functionality?
            max_depth_options['categories'] = range(int(np.rint(np.exp(0))), int(np.rint(np.exp(5.0))))

            min_samples_split_options = OrderedDict()
            min_samples_split_options['distribution'] = 'Uniform'
            min_samples_split_options['lower'] = 2
            min_samples_split_options['upper'] = 20
            min_samples_split_options['categories'] = range(2, 20)

            min_samples_leaf_options = OrderedDict()
            min_samples_leaf_options['distribution'] = 'Uniform'
            min_samples_leaf_options['lower'] = 1
            min_samples_leaf_options['upper'] = 20
            min_samples_leaf_options['categories'] = range(1, 20)

            learner_options = {'criterion': criterion_options,
                               'max_features_use_preset': max_features_use_preset_options,
                               'max_features_preset': max_features_preset_options,
                               'max_features_sample': max_features_sample_options,
                               'max_depth': max_depth_options,
                               'min_samples_split': min_samples_split_options,
                               'min_samples_leaf': min_samples_leaf_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['DecisionTreeClassifier'][optimizer]

        # Build priors
        criterion_prior = Distribution.load(learner_parameters['criterion'])
        max_features_use_preset_prior = Distribution.load(learner_parameters['max_features_use_preset'], category_type='bool')
        max_features_preset_prior = Distribution.load(learner_parameters['max_features_preset'], parse_none=True)
        max_features_sample_prior = Distribution.load(learner_parameters['max_features_sample'])
        max_depth_prior = Distribution.load(learner_parameters['max_depth'], category_type='int')
        min_samples_split_prior = Distribution.load(learner_parameters['min_samples_split'], category_type='int')
        min_samples_leaf_prior = Distribution.load(learner_parameters['min_samples_leaf'], category_type='int')

        # Build parameters
        criterion_param = Parameter('criterion', prior=criterion_prior, default='gini')
        max_features_use_preset_param = Parameter('max_features_use_preset',
                                                  prior=max_features_use_preset_prior,
                                                  default=True)
        max_features_preset_param = Parameter('max_features_preset',
                                              prior=max_features_preset_prior,
                                              default=None,
                                              sample_if=lambda params: params['max_features_use_preset'])
        max_features_sample_param = Parameter('max_features_sample',
                                              prior=max_features_sample_prior,
                                              default=1.0,
                                              sample_if=lambda params: not params['max_features_use_preset'])
        max_depth_param = Parameter('max_depth', prior=max_depth_prior, default=None)
        min_samples_split_param = Parameter('min_samples_split', prior=min_samples_split_prior, default=2)
        min_samples_leaf_param = Parameter('min_samples_leaf', prior=min_samples_leaf_prior, default=1)

        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['criterion'] = criterion_param
        param_space['max_features_use_preset'] = max_features_use_preset_param
        param_space['max_features_preset'] = max_features_preset_param
        param_space['max_features_sample'] = max_features_sample_param
        param_space['max_depth'] = max_depth_param
        param_space['min_samples_split'] = min_samples_split_param
        param_space['min_samples_leaf'] = min_samples_leaf_param
        param_space['random_state'] = random_state_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['DecisionTreeClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['criterion'] = param.CategoricalParameter('criterion', ['gini', 'entropy'], default='gini')
        param_space['max_features_use_preset'] = param.CategoricalParameter('max_features_use_preset', [True, False], default=False)
        param_max_features_preset = param.CategoricalParameter('max_features_preset', ['sqrt', 'log2', None], default=None)
        param_space['max_features_use_preset'].categories[True]['max_features_preset'] = param_max_features_preset
        param_max_features_sample = param.NumericalParameter('max_features_sample', prior=param.Uniform(0.0, 1.0), default=1.0)
        param_space['max_features_use_preset'].categories[False]['max_features_sample'] = param_max_features_sample

        param_space['max_depth'] = param.NumericalParameter('max_depth', prior=param.LogUniform(lower=0.0, upper=5.0), default=None, discretize=True)
        param_space['min_samples_split'] = param.NumericalParameter('min_samples_split', prior=param.Uniform(lower=2, upper=20), default=2, discretize=True)
        param_space['min_samples_leaf'] = param.NumericalParameter('min_samples_leaf', prior=param.Uniform(lower=1, upper=20), default=1, discretize=True)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Common numerical hyperparameters
        param_space['max_depth'] = param.NumericalParameter('max_depth', prior=param.LogUniform(lower=0.0, upper=5.0), default=None, discretize=True)
        param_space['min_samples_split'] = param.NumericalParameter('min_samples_split', prior=param.Uniform(lower=2, upper=20), default=2, discretize=True)
        param_space['min_samples_leaf'] = param.NumericalParameter('min_samples_leaf', prior=param.Uniform(lower=1, upper=20), default=1, discretize=True)

        criterion_cats = ['gini', 'entropy']
        param_space['criterion'] = param.CategoricalParameter('criterion', criterion_cats, default='gini')
        for criterion in criterion_cats:
            max_features_use_preset_cats = [True, False]
            criterion_branch = param_space['criterion'].categories[criterion]
            criterion_branch['max_features_use_preset'] = param.CategoricalParameter('max_features_use_preset', max_features_use_preset_cats, default=False)
            max_features_use_preset = criterion_branch['max_features_use_preset'].categories[True]
            max_features_use_preset['max_features_preset'] = param.CategoricalParameter('max_features_preset', ['sqrt', 'log2', None], default=None)
            max_features_dont_use_preset = criterion_branch['max_features_use_preset'].categories[False]
            max_features_dont_use_preset['max_features_sample'] = param.NumericalParameter('max_features_sample', prior=param.Uniform(0.0, 1.0), default=1.0)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class ExtraTreeClassifier(BaseClassifier):
    """Extremely randomized tree classifier."""

    # Removed; see http://stackoverflow.com/questions/20177970/decisiontreeclassifier-vs-extratreeclassifier

    def create_classifier(self, **parameters):
        """Creates the extremely randomized tree classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Extremely randomized decision tree classifier.
        """
        if 'max_features_use_preset' in parameters:
            del parameters['max_features_use_preset']
            if 'max_features_preset' in parameters:
                parameters['max_features'] = parameters['max_features_preset']
                del parameters['max_features_preset']
            else:
                parameters['max_features'] = parameters['max_features_sample']
                del parameters['max_features_sample']
        classifier = SKExtraTreeClassifier(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['criterion'] = ChoiceParameter(['gini', 'entropy'])
        param_space['splitter'] = ChoiceParameter(['best', 'random'])
        param_space['max_depth'] = ChoiceParameter([3])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        #param_space['max_features'] = ?
        #param_space['random_state'] = ?

        return param_space


class RandomForestClassifier(BaseClassifier):
    """Random forest classifier."""

    def create_classifier(self, **parameters):
        """Creates the random forest classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Random forest classifier.
        """
        if 'max_features_use_preset' in parameters:
            del parameters['max_features_use_preset']
            if 'max_features_preset' in parameters:
                parameters['max_features'] = parameters['max_features_preset']
                del parameters['max_features_preset']
            else:
                parameters['max_features'] = parameters['max_features_sample']
                del parameters['max_features_sample']
        classifier = SKRandomForestClassifier(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_estimators'] = ChoiceParameter([10])
        param_space['criterion'] = ChoiceParameter(['gini', 'entropy'])
        #param_space['max_features'] = ?
        param_space['max_depth'] = ChoiceParameter([3])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        #param_space['bootstrap'] = BooleanParameter()  # must be True for oob scores=True to be valid
        param_space['oob_score'] = BooleanParameter()
        #param_space['random_state'] = ?

        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            n_estimators_options = OrderedDict()
            n_estimators_options['distribution'] = 'Categorical'
            n_estimators_options['categories'] = range(2, 20)

            criterion_options = OrderedDict()
            criterion_options['distribution'] = 'Categorical'
            criterion_options['categories'] = ['gini', 'entropy']
            criterion_options['probabilities'] = [0.5, 0.5]

            max_features_use_preset_options = OrderedDict()
            max_features_use_preset_options['distribution'] = 'Categorical'
            max_features_use_preset_options['categories'] = [True, False]
            max_features_use_preset_options['probabilities'] = [0.75, 0.25]

            max_features_preset_options = OrderedDict()
            max_features_preset_options['distribution'] = 'Categorical'
            max_features_preset_options['categories'] = ['sqrt', 'log2', None]

            max_features_sample_options = OrderedDict()
            max_features_sample_options['distribution'] = 'Uniform'
            max_features_sample_options['lower'] = 0.0
            max_features_sample_options['upper'] = 1.0

            max_depth_options = OrderedDict()
            max_depth_options['distribution'] = 'LogUniform'
            max_depth_options['lower'] = 0.0
            max_depth_options['upper'] = 5.0
            # TODO add discretization functionality?
            max_depth_options['categories'] = range(int(np.rint(np.exp(0))), int(np.rint(np.exp(5.0))))

            min_samples_split_options = OrderedDict()
            min_samples_split_options['distribution'] = 'Uniform'
            min_samples_split_options['lower'] = 2
            min_samples_split_options['upper'] = 20
            min_samples_split_options['categories'] = range(2, 20)

            min_samples_leaf_options = OrderedDict()
            min_samples_leaf_options['distribution'] = 'Uniform'
            min_samples_leaf_options['lower'] = 1
            min_samples_leaf_options['upper'] = 20
            min_samples_leaf_options['categories'] = range(1, 20)

            bootstrap_options = OrderedDict()
            bootstrap_options['distribution'] = 'Categorical'
            bootstrap_options['categories'] = [True, False]

            oob_options = OrderedDict()
            oob_options['distribution'] = 'Categorical'
            oob_options['categories'] = [True, False]

            learner_options = {'n_estimators': n_estimators_options,
                               'criterion': criterion_options,
                               'max_features_use_preset': max_features_use_preset_options,
                               'max_features_preset': max_features_preset_options,
                               'max_features_sample': max_features_sample_options,
                               'max_depth': max_depth_options,
                               'min_samples_split': min_samples_split_options,
                               'min_samples_leaf': min_samples_leaf_options,
                               'bootstrap': bootstrap_options,
                               'oob_score': oob_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['RandomForestClassifier'][optimizer]

        # Build priors
        n_estimators_prior = Distribution.load(learner_parameters['n_estimators'], category_type='int')
        criterion_prior = Distribution.load(learner_parameters['criterion'])
        max_features_use_preset_prior = Distribution.load(learner_parameters['max_features_use_preset'], category_type='bool')
        max_features_preset_prior = Distribution.load(learner_parameters['max_features_preset'], parse_none=True)
        max_features_sample_prior = Distribution.load(learner_parameters['max_features_sample'])
        max_depth_prior = Distribution.load(learner_parameters['max_depth'], category_type='int')
        min_samples_split_prior = Distribution.load(learner_parameters['min_samples_split'], category_type='int')
        min_samples_leaf_prior = Distribution.load(learner_parameters['min_samples_leaf'], category_type='int')
        bootstrap_prior = Distribution.load(learner_parameters['bootstrap'], category_type='bool')
        oob_score_prior = Distribution.load(learner_parameters['oob_score'], category_type='bool')

        # Build parameters
        n_estimators_param = Parameter('n_estimators', prior=n_estimators_prior, default=10)
        criterion_param = Parameter('criterion', prior=criterion_prior, default='gini')
        max_features_use_preset_param = Parameter('max_features_use_preset',
                                                  prior=max_features_use_preset_prior,
                                                  default=True)
        max_features_preset_param = Parameter('max_features_preset',
                                              prior=max_features_preset_prior,
                                              default=None,
                                              sample_if=lambda params: params['max_features_use_preset'])
        max_features_sample_param = Parameter('max_features_sample',  # Just let it break when feature percentage rounds down to zero?
                                              prior=max_features_sample_prior,
                                              default=1.0,
                                              sample_if=lambda params: not params['max_features_use_preset'])
        max_depth_param = Parameter('max_depth', prior=max_depth_prior, default=None)
        min_samples_split_param = Parameter('min_samples_split', prior=min_samples_split_prior, default=2)
        min_samples_leaf_param = Parameter('min_samples_leaf', prior=min_samples_leaf_prior, default=1)
        bootstrap_param = Parameter('bootstrap', prior=bootstrap_prior, default=False)
        oob_score_param = Parameter('oob_score', prior=oob_score_prior, default=False,
                                    sample_if=lambda params: params['bootstrap'])

        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['n_estimators'] = n_estimators_param
        param_space['criterion'] = criterion_param
        param_space['max_features_use_preset'] = max_features_use_preset_param
        param_space['max_features_preset'] = max_features_preset_param
        param_space['max_features_sample'] = max_features_sample_param
        param_space['max_depth'] = max_depth_param
        param_space['min_samples_split'] = min_samples_split_param
        param_space['min_samples_leaf'] = min_samples_leaf_param
        param_space['bootstrap'] = bootstrap_param
        param_space['oob_score'] = oob_score_param
        param_space['random_state'] = random_state_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['RandomForestClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['criterion'] = param.CategoricalParameter('criterion', ['gini', 'entropy'], default='gini')
        param_space['max_features_use_preset'] = param.CategoricalParameter('max_features_use_preset', [True, False], default=False)
        param_max_features_preset = param.CategoricalParameter('max_features_preset', ['sqrt', 'log2', None], default=None)
        param_space['max_features_use_preset'].categories[True]['max_features_preset'] = param_max_features_preset
        param_max_features_sample = param.NumericalParameter('max_features_sample', prior=param.Uniform(0.0, 1.0), default=1.0)
        param_space['max_features_use_preset'].categories[False]['max_features_sample'] = param_max_features_sample
        param_space['bootstrap'] = param.CategoricalParameter('bootstrap', [True, False], default=True)
        param_space['bootstrap'].categories[True]['oob_score'] = param.CategoricalParameter('oob_score', [True, False], default=False)

        param_space['n_estimators'] = param.NumericalParameter('n_estimators', prior=param.Uniform(lower=2, upper=20), default=10, discretize=True)
        param_space['max_depth'] = param.NumericalParameter('max_depth', prior=param.LogUniform(lower=0.0, upper=5.0), default=None, discretize=True)
        param_space['min_samples_split'] = param.NumericalParameter('min_samples_split', prior=param.Uniform(lower=2, upper=20), default=2, discretize=True)
        param_space['min_samples_leaf'] = param.NumericalParameter('min_samples_leaf', prior=param.Uniform(lower=1, upper=20), default=1, discretize=True)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Common numerical hyperparameters
        param_space['n_estimators'] = param.NumericalParameter('n_estimators', prior=param.Uniform(lower=2, upper=20), default=10, discretize=True)
        param_space['max_depth'] = param.NumericalParameter('max_depth', prior=param.LogUniform(lower=0.0, upper=5.0), default=None, discretize=True)
        param_space['min_samples_split'] = param.NumericalParameter('min_samples_split', prior=param.Uniform(lower=2, upper=20), default=2, discretize=True)
        param_space['min_samples_leaf'] = param.NumericalParameter('min_samples_leaf', prior=param.Uniform(lower=1, upper=20), default=1, discretize=True)

        bootstrap_branch = param.CategoricalParameter('bootstrap', [True, False], default=True)
        bootstrap_branch.categories[True]['oob_score'] = param.CategoricalParameter('oob_score', [True, False], default=False)

        criterion_cats = ['gini', 'entropy']
        param_space['criterion'] = param.CategoricalParameter('criterion', criterion_cats, default='gini')
        for criterion in criterion_cats:
            max_features_use_preset_cats = [True, False]
            criterion_branch = param_space['criterion'].categories[criterion]
            criterion_branch['max_features_use_preset'] = param.CategoricalParameter('max_features_use_preset', max_features_use_preset_cats, default=False)
            max_features_use_preset = criterion_branch['max_features_use_preset'].categories[True]
            preset_cats = ['sqrt', 'log2', None]
            max_features_use_preset['max_features_preset'] = param.CategoricalParameter('max_features_preset', preset_cats, default=None)
            for preset in preset_cats:
                preset_branch = max_features_use_preset['max_features_preset'].categories[preset]
                preset_branch['bootstrap'] = bootstrap_branch
            max_features_dont_use_preset = criterion_branch['max_features_use_preset'].categories[False]
            max_features_dont_use_preset['max_features_sample'] = param.NumericalParameter('max_features_sample', prior=param.Uniform(0.0, 1.0), default=1.0)
            max_features_dont_use_preset['bootstrap'] = bootstrap_branch

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class ExtraTreeEnsembleClassifier(BaseClassifier):
    """Classifier based on a number of randomized decision trees."""

    def create_classifier(self, **parameters):
        """Creates the extra trees class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Extra trees classifier.
        """
        if 'max_features_use_preset' in parameters:
            del parameters['max_features_use_preset']
            if 'max_features_preset' in parameters:
                parameters['max_features'] = parameters['max_features_preset']
                del parameters['max_features_preset']
            else:
                parameters['max_features'] = parameters['max_features_sample']
                del parameters['max_features_sample']
        classifier = SKExtraTreesClassifier(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_estimators'] = ChoiceParameter([10])
        param_space['criterion'] = ChoiceParameter(['gini', 'entropy'])
        #param_space['max_features'] = ?
        param_space['max_depth'] = ChoiceParameter([3])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        #param_space['bootstrap'] = BooleanParameter() see RandomForestClassifier
        param_space['bootstrap'] = ChoiceParameter([True])
        param_space['oob_score'] = BooleanParameter()
        #param_space['random_state'] = ?

        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            n_estimators_options = OrderedDict()
            n_estimators_options['distribution'] = 'Categorical'
            n_estimators_options['categories'] = range(2, 20)

            criterion_options = OrderedDict()
            criterion_options['distribution'] = 'Categorical'
            criterion_options['categories'] = ['gini', 'entropy']
            criterion_options['probabilities'] = [0.5, 0.5]

            max_features_use_preset_options = OrderedDict()
            max_features_use_preset_options['distribution'] = 'Categorical'
            max_features_use_preset_options['categories'] = [True, False]
            max_features_use_preset_options['probabilities'] = [0.75, 0.25]

            max_features_preset_options = OrderedDict()
            max_features_preset_options['distribution'] = 'Categorical'
            max_features_preset_options['categories'] = ['sqrt', 'log2', None]

            max_features_sample_options = OrderedDict()
            max_features_sample_options['distribution'] = 'Uniform'
            max_features_sample_options['lower'] = 0.0
            max_features_sample_options['upper'] = 1.0

            max_depth_options = OrderedDict()
            max_depth_options['distribution'] = 'LogUniform'
            max_depth_options['lower'] = 0.0
            max_depth_options['upper'] = 5.0
            # TODO add discretization functionality?
            max_depth_options['categories'] = range(int(np.rint(np.exp(0))), int(np.rint(np.exp(5.0))))

            min_samples_split_options = OrderedDict()
            min_samples_split_options['distribution'] = 'Uniform'
            min_samples_split_options['lower'] = 2
            min_samples_split_options['upper'] = 20
            min_samples_split_options['categories'] = range(2, 20)

            min_samples_leaf_options = OrderedDict()
            min_samples_leaf_options['distribution'] = 'Uniform'
            min_samples_leaf_options['lower'] = 1
            min_samples_leaf_options['upper'] = 20
            min_samples_leaf_options['categories'] = range(1, 20)

            bootstrap_options = OrderedDict()
            bootstrap_options['distribution'] = 'Categorical'
            bootstrap_options['categories'] = [True, False]

            oob_options = OrderedDict()
            oob_options['distribution'] = 'Categorical'
            oob_options['categories'] = [True, False]

            learner_options = {'n_estimators': n_estimators_options,
                               'criterion': criterion_options,
                               'max_features_use_preset': max_features_use_preset_options,
                               'max_features_preset': max_features_preset_options,
                               'max_features_sample': max_features_sample_options,
                               'max_depth': max_depth_options,
                               'min_samples_split': min_samples_split_options,
                               'min_samples_leaf': min_samples_leaf_options,
                               'bootstrap': bootstrap_options,
                               'oob_score': oob_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['ExtraTreeEnsembleClassifier'][optimizer]

        # Build priors
        n_estimators_prior = Distribution.load(learner_parameters['n_estimators'], category_type='int')
        criterion_prior = Distribution.load(learner_parameters['criterion'])
        max_features_use_preset_prior = Distribution.load(learner_parameters['max_features_use_preset'], category_type='bool')
        max_features_preset_prior = Distribution.load(learner_parameters['max_features_preset'], parse_none=True)
        max_features_sample_prior = Distribution.load(learner_parameters['max_features_sample'])
        max_depth_prior = Distribution.load(learner_parameters['max_depth'], category_type='int')
        min_samples_split_prior = Distribution.load(learner_parameters['min_samples_split'], category_type='int')
        min_samples_leaf_prior = Distribution.load(learner_parameters['min_samples_leaf'], category_type='int')
        bootstrap_prior = Distribution.load(learner_parameters['bootstrap'], category_type='bool')
        oob_score_prior = Distribution.load(learner_parameters['oob_score'], category_type='bool')

        # Build parameters
        n_estimators_param = Parameter('n_estimators', prior=n_estimators_prior, default=10)
        criterion_param = Parameter('criterion', prior=criterion_prior, default='gini')
        max_features_use_preset_param = Parameter('max_features_use_preset',
                                                  prior=max_features_use_preset_prior,
                                                  default=True)
        max_features_preset_param = Parameter('max_features_preset',
                                              prior=max_features_preset_prior,
                                              default=None,
                                              sample_if=lambda params: params['max_features_use_preset'])
        max_features_sample_param = Parameter('max_features_sample',
                                              prior=max_features_sample_prior,
                                              default=1.0,
                                              sample_if=lambda params: not params['max_features_use_preset'])
        max_depth_param = Parameter('max_depth', prior=max_depth_prior, default=None)
        min_samples_split_param = Parameter('min_samples_split', prior=min_samples_split_prior, default=2)
        min_samples_leaf_param = Parameter('min_samples_leaf', prior=min_samples_leaf_prior, default=1)
        bootstrap_param = Parameter('bootstrap', prior=bootstrap_prior, default=False)
        oob_score_param = Parameter('oob_score', prior=oob_score_prior, default=False,
                                    sample_if=lambda params: params['bootstrap'])

        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['n_estimators'] = n_estimators_param
        param_space['criterion'] = criterion_param
        param_space['max_features_use_preset'] = max_features_use_preset_param
        param_space['max_features_preset'] = max_features_preset_param
        param_space['max_features_sample'] = max_features_sample_param
        param_space['max_depth'] = max_depth_param
        param_space['min_samples_split'] = min_samples_split_param
        param_space['min_samples_leaf'] = min_samples_leaf_param
        param_space['bootstrap'] = bootstrap_param
        param_space['oob_score'] = oob_score_param
        param_space['random_state'] = random_state_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['ExtraTreeEnsembleClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['criterion'] = param.CategoricalParameter('criterion', ['gini', 'entropy'], default='gini')
        param_space['max_features_use_preset'] = param.CategoricalParameter('max_features_use_preset', [True, False], default=False)
        param_max_features_preset = param.CategoricalParameter('max_features_preset', ['sqrt', 'log2', None], default=None)
        param_space['max_features_use_preset'].categories[True]['max_features_preset'] = param_max_features_preset
        param_max_features_sample = param.NumericalParameter('max_features_sample', prior=param.Uniform(0.0, 1.0), default=1.0)
        param_space['max_features_use_preset'].categories[False]['max_features_sample'] = param_max_features_sample
        param_space['bootstrap'] = param.CategoricalParameter('bootstrap', [True, False], default=True)
        param_space['bootstrap'].categories[True]['oob_score'] = param.CategoricalParameter('oob_score', [True, False], default=False)

        param_space['n_estimators'] = param.NumericalParameter('n_estimators', prior=param.Uniform(lower=2, upper=20), default=10, discretize=True)
        param_space['max_depth'] = param.NumericalParameter('max_depth', prior=param.LogUniform(lower=0.0, upper=5.0), default=None, discretize=True)
        param_space['min_samples_split'] = param.NumericalParameter('min_samples_split', prior=param.Uniform(lower=2, upper=20), default=2, discretize=True)
        param_space['min_samples_leaf'] = param.NumericalParameter('min_samples_leaf', prior=param.Uniform(lower=1, upper=20), default=1, discretize=True)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Common numerical hyperparameters
        param_space['n_estimators'] = param.NumericalParameter('n_estimators', prior=param.Uniform(lower=2, upper=20), default=10, discretize=True)
        param_space['max_depth'] = param.NumericalParameter('max_depth', prior=param.LogUniform(lower=0.0, upper=5.0), default=None, discretize=True)
        param_space['min_samples_split'] = param.NumericalParameter('min_samples_split', prior=param.Uniform(lower=2, upper=20), default=2, discretize=True)
        param_space['min_samples_leaf'] = param.NumericalParameter('min_samples_leaf', prior=param.Uniform(lower=1, upper=20), default=1, discretize=True)

        bootstrap_branch = param.CategoricalParameter('bootstrap', [True, False], default=False)
        bootstrap_branch.categories[True]['oob_score'] = param.CategoricalParameter('oob_score', [True, False], default=False)

        criterion_cats = ['gini', 'entropy']
        param_space['criterion'] = param.CategoricalParameter('criterion', criterion_cats, default='gini')
        for criterion in criterion_cats:
            max_features_use_preset_cats = [True, False]
            criterion_branch = param_space['criterion'].categories[criterion]
            criterion_branch['max_features_use_preset'] = param.CategoricalParameter('max_features_use_preset', max_features_use_preset_cats, default=False)
            max_features_use_preset = criterion_branch['max_features_use_preset'].categories[True]
            preset_cats = ['sqrt', 'log2', None]
            max_features_use_preset['max_features_preset'] = param.CategoricalParameter('max_features_preset', preset_cats, default=None)
            for preset in preset_cats:
                preset_branch = max_features_use_preset['max_features_preset'].categories[preset]
                preset_branch['bootstrap'] = bootstrap_branch
            max_features_dont_use_preset = criterion_branch['max_features_use_preset'].categories[False]
            max_features_dont_use_preset['max_features_sample'] = param.NumericalParameter('max_features_sample', prior=param.Uniform(0.0, 1.0), default=1.0)
            max_features_dont_use_preset['bootstrap'] = bootstrap_branch

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class GradientBoostingClassifier(BaseClassifier):
    """Gradient boosting classifier."""

    def create_classifier(self, **parameters):
        """Creates the gradient boosting class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Gradient boosting classifier.
        """
        if 'max_features_use_preset' in parameters:
            del parameters['max_features_use_preset']
            if 'max_features_preset' in parameters:
                parameters['max_features'] = parameters['max_features_preset']
                del parameters['max_features_preset']
            else:
                parameters['max_features'] = parameters['max_features_sample']
                del parameters['max_features_sample']
        classifier = SKGradientBoostingClassifier(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()
        """
        #        param_space['loss'] = ChoiceParameter(['deviance', 'ls', 'lad', 'huber', 'quantile',
        #                                               'bdeviance', 'mdeviance'])
        # bdeviance only valid for two classes
        # for classification, only 'deviance' is allowed (other loss functions are
        # valid for regression only)
                #param_space['loss'] = ChoiceParameter(['deviance', 'ls', 'lad', 'huber', 'quantile',
                #                                       'mdeviance'])
        """
        param_space['loss'] = ChoiceParameter(['deviance', 'mdeviance'])
        param_space['learning_rate'] = ChoiceParameter([0.1])
        param_space['n_estimators'] = ChoiceParameter([100])
        param_space['max_depth'] = ChoiceParameter([3])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        param_space['subsample'] = ChoiceParameter([1.0])  # ?
        # param_space['max_features'] = ?
        # param_space['init'] = ?

        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            loss_options = OrderedDict()
            loss_options['distribution'] = 'Categorical'
            loss_options['categories'] = ['deviance']  # Only one loss?

            learning_rate_options = OrderedDict()
            learning_rate_options['distribution'] = 'Uniform'
            learning_rate_options['lower'] = 0.0
            learning_rate_options['upper'] = 1.0

            n_estimators_options = OrderedDict()
            n_estimators_options['distribution'] = 'LogUniform'
            n_estimators_options['lower'] = 2.3
            n_estimators_options['upper'] = 6.5
            #n_estimators_options['discretize'] = 'round'

            max_depth_options = OrderedDict()
            max_depth_options['distribution'] = 'Uniform'
            max_depth_options['lower'] = 2.0
            max_depth_options['upper'] = 10.0
            # TODO add discretization functionality?
            #max_depth_options['discretize'] = 'round'

            min_samples_split_options = OrderedDict()
            min_samples_split_options['distribution'] = 'Uniform'
            min_samples_split_options['lower'] = 2
            min_samples_split_options['upper'] = 20
            #min_samples_split_options['discretize'] = 'round'

            min_samples_leaf_options = OrderedDict()
            min_samples_leaf_options['distribution'] = 'Uniform'
            min_samples_leaf_options['lower'] = 1
            min_samples_leaf_options['upper'] = 20
            #min_samples_leaf_options['discretize'] = 'round'

            subsample_options = OrderedDict()
            subsample_options['distribution'] = 'Uniform'
            subsample_options['lower'] = 0.0
            subsample_options['upper'] = 1.0

            max_features_use_preset_options = OrderedDict()
            max_features_use_preset_options['distribution'] = 'Categorical'
            max_features_use_preset_options['categories'] = [True, False]
            max_features_use_preset_options['probabilities'] = [0.75, 0.25]

            max_features_preset_options = OrderedDict()
            max_features_preset_options['distribution'] = 'Categorical'
            max_features_preset_options['categories'] = ['sqrt', 'log2', None]

            max_features_sample_options = OrderedDict()
            max_features_sample_options['distribution'] = 'Uniform'
            max_features_sample_options['lower'] = 0.0
            max_features_sample_options['upper'] = 1.0

            learner_options = {'loss': loss_options,
                               'learning_rate': learning_rate_options,
                               'n_estimators': n_estimators_options,
                               'max_depth': max_depth_options,
                               'min_samples_split': min_samples_split_options,
                               'min_samples_leaf': min_samples_leaf_options,
                               'subsample': subsample_options,
                               'max_features_use_preset': max_features_use_preset_options,
                               'max_features_preset': max_features_preset_options,
                               'max_features_sample': max_features_sample_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['GradientBoostingClassifier'][optimizer]

        # Build priors
        loss_prior = Distribution.load(learner_parameters['loss'])
        learning_rate_prior = Distribution.load(learner_parameters['learning_rate'])
        n_estimators_prior = Distribution.load(learner_parameters['n_estimators'])
        max_depth_prior = Distribution.load(learner_parameters['max_depth'])
        min_samples_split_prior = Distribution.load(learner_parameters['min_samples_split'])
        min_samples_leaf_prior = Distribution.load(learner_parameters['min_samples_leaf'])
        subsample_prior = Distribution.load(learner_parameters['subsample'])
        max_features_use_preset_prior = Distribution.load(learner_parameters['max_features_use_preset'], category_type='bool')
        max_features_preset_prior = Distribution.load(learner_parameters['max_features_preset'], parse_none=True)
        max_features_sample_prior = Distribution.load(learner_parameters['max_features_sample'])

        # Build parameters
        loss_param = Parameter('loss', prior=loss_prior, default='deviance')
        learning_rate_param = Parameter('learning_rate', prior=learning_rate_prior, default=0.1)
        n_estimators_param = Parameter('n_estimators', prior=n_estimators_prior, default=100, discretize='round')
        max_depth_param = Parameter('max_depth', prior=max_depth_prior, default=3, discretize='round')
        min_samples_split_param = Parameter('min_samples_split', prior=min_samples_split_prior, default=2, discretize='round')
        min_samples_leaf_param = Parameter('min_samples_leaf', prior=min_samples_leaf_prior, default=1, discretize='round')
        subsample_param = Parameter('subsample', prior=subsample_prior, default=1.0)
        max_features_use_preset_param = Parameter('max_features_use_preset',
                                                  prior=max_features_use_preset_prior,
                                                  default=True)
        max_features_preset_param = Parameter('max_features_preset',
                                              prior=max_features_preset_prior,
                                              default=None,
                                              sample_if=lambda params: params['max_features_use_preset'])
        max_features_sample_param = Parameter('max_features_sample',
                                              prior=max_features_sample_prior,
                                              default=1.0,
                                              sample_if=lambda params: not params['max_features_use_preset'])
        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['loss'] = loss_param
        param_space['learning_rate'] = learning_rate_param
        param_space['n_estimators'] = n_estimators_param
        param_space['max_depth'] = max_depth_param
        param_space['max_features_use_preset'] = max_features_use_preset_param
        param_space['min_samples_split'] = min_samples_split_param
        param_space['min_samples_leaf'] = min_samples_leaf_param
        param_space['subsample'] = subsample_param
        param_space['max_features_preset'] = max_features_preset_param
        param_space['max_features_sample'] = max_features_sample_param
        param_space['random_state'] = random_state_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['GradientBoostingClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['loss'] = param.CategoricalParameter('loss', ['deviance'], default='deviance')
        param_space['max_features_use_preset'] = param.CategoricalParameter('max_features_use_preset', [True, False], default=False)
        param_max_features_preset = param.CategoricalParameter('max_features_preset', ['sqrt', 'log2', None], default=None)
        param_space['max_features_use_preset'].categories[True]['max_features_preset'] = param_max_features_preset
        param_max_features_sample = param.NumericalParameter('max_features_sample', prior=param.Uniform(0.0, 1.0), default=1.0)
        param_space['max_features_use_preset'].categories[False]['max_features_sample'] = param_max_features_sample

        param_space['learning_rate'] = param.NumericalParameter('learning_rate', prior=param.Uniform(lower=0.0, upper=1.0), default=0.1)
        param_space['n_estimators'] = param.NumericalParameter('n_estimators', prior=param.Uniform(lower=2, upper=20), default=10, discretize=True)
        param_space['max_depth'] = param.NumericalParameter('max_depth', prior=param.LogUniform(lower=0.0, upper=5.0), default=None, discretize=True)
        param_space['min_samples_split'] = param.NumericalParameter('min_samples_split', prior=param.Uniform(lower=2, upper=20), default=2, discretize=True)
        param_space['min_samples_leaf'] = param.NumericalParameter('min_samples_leaf', prior=param.Uniform(lower=1, upper=20), default=1, discretize=True)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['loss'] = param.CategoricalParameter('loss', ['deviance'], default='deviance')
        loss_branch = param_space['loss'].categories['deviance']
        loss_branch['max_features_use_preset'] = param.CategoricalParameter('max_features_use_preset', [True, False], default=False)
        param_max_features_preset = param.CategoricalParameter('max_features_preset', ['sqrt', 'log2', None], default=None)
        loss_branch['max_features_use_preset'].categories[True]['max_features_preset'] = param_max_features_preset
        param_max_features_sample = param.NumericalParameter('max_features_sample', prior=param.Uniform(0.0, 1.0), default=1.0)
        loss_branch['max_features_use_preset'].categories[False]['max_features_sample'] = param_max_features_sample

        param_space['learning_rate'] = param.NumericalParameter('learning_rate', prior=param.Uniform(lower=0.0, upper=1.0), default=0.1)
        param_space['n_estimators'] = param.NumericalParameter('n_estimators', prior=param.Uniform(lower=2, upper=20), default=10, discretize=True)
        param_space['max_depth'] = param.NumericalParameter('max_depth', prior=param.LogUniform(lower=0.0, upper=5.0), default=None, discretize=True)
        param_space['min_samples_split'] = param.NumericalParameter('min_samples_split', prior=param.Uniform(lower=2, upper=20), default=2, discretize=True)
        param_space['min_samples_leaf'] = param.NumericalParameter('min_samples_leaf', prior=param.Uniform(lower=1, upper=20), default=1, discretize=True)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class GaussianProcessClassifier(BaseClassifier):
    """Gaussian Process classifier."""

    def create_classifier(self, **parameters):
        """Creates the Gaussian process classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Gaussian process classifier.
        """
        if 'estimate_ml' in parameters:
            if not parameters['estimate_ml']:
                #del parameters['thetaL']
                #del parameters['thetaU']
                parameters['thetaL'] = None
                parameters['thetaU'] = None
            else:
                parameters['thetaL'] = np.array([parameters['thetaL']])
                parameters['thetaU'] = np.array([parameters['thetaU']])
            del parameters['estimate_ml']
        # p parameter needed for correlation = 'generalized_exponential'
        '''
        if 'p' in parameters:
            parameters['theta0'] = np.array([parameters['theta0'], parameters['p']])
            print(parameters['theta0'])
            if 'thetaL' in parameters:
                parameters['thetaL'] = np.array([parameters['thetaL'], parameters['p']])
            if 'thetaU' in parameters:
                parameters['thetaU'] = np.array([parameters['thetaU'], parameters['p']])
            del parameters['p']
        '''
        classifier = GaussianProcess(**parameters)
        return classifier

    #@log_step("Predicting with Gaussian process classifier")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        classifier = self.classifier
        prediction = classifier.predict(data)
        prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['regr'] = ChoiceParameter(['constant', 'linear', 'quadratic'])
        param_space['corr'] = ChoiceParameter(['absolute_exponential', 'squared_exponential',
                                               'generalized_exponential', 'cubic', 'linear'])
        # param_space['beta0'] = ?
        param_space['storage_mode'] = ChoiceParameter(['full', 'light'])
        # param_space['theta0'] = ?
        # param_space['thetaL'] = ?
        # param_space['thetaU'] = ?
        param_space['normalize'] = BooleanParameter()
        #param_space['nugget'] = BooleanParameter() ???
        param_space['optimizer'] = ChoiceParameter(['fmin_cobyla', 'Welch'])
        param_space['random_start'] = ChoiceParameter([1])

        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            regr_options = OrderedDict()
            regr_options['distribution'] = 'Categorical'
            regr_options['categories'] = ['constant', 'linear', 'quadratic']  # Only one loss?

            corr_options = OrderedDict()
            corr_options['distribution'] = 'Categorical'
            corr_options['categories'] = ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear']

            #beta0 not modeled

            theta0_options = OrderedDict()
            theta0_options['distribution'] = 'Uniform'  # LogNormal?
            theta0_options['lower'] = 0.0
            theta0_options['upper'] = 1.0

            estimate_ml_options = OrderedDict()
            estimate_ml_options['distribution'] = 'Categorical'
            estimate_ml_options['categories'] = [True, False]
            estimate_ml_options['probabilities'] = [0.75, 0.25]

            thetaL_options = OrderedDict()
            thetaL_options['distribution'] = 'Uniform'  # LogNormal?
            thetaL_options['lower'] = 0.0
            thetaL_options['upper'] = 1.0

            thetaU_options = OrderedDict()
            thetaU_options['distribution'] = 'Uniform'  # LogNormal?
            thetaU_options['lower'] = 0.0
            thetaU_options['upper'] = 1.0

            normalize_options = OrderedDict()
            normalize_options['distribution'] = 'Categorical'
            normalize_options['categories'] = [True, False]
            normalize_options['probabilities'] = [0.75, 0.25]

            nugget_options = OrderedDict()
            nugget_options['distribution'] = 'uniform'
            nugget_options['lower'] = 5 * np.finfo(float).eps
            nugget_options['upper'] = 1.0  # ?

            optimizer_options = OrderedDict()
            optimizer_options['distribution'] = 'Categorical'
            optimizer_options['categories'] = ['fmin_cobyla', 'Welch']
            optimizer_options['probabilities'] = [0.5, 0.5]

            learner_options = {'regr': regr_options,
                               'corr': corr_options,
                               'theta0': theta0_options,
                               'estimate_ml': estimate_ml_options,
                               'thetaL': thetaL_options,
                               'thetaU': thetaU_options,
                               'normalize': normalize_options,
                               'nugget': nugget_options,
                               'optimizer': optimizer_options,
                               'enabled': False}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['GaussianProcessClassifier'][optimizer]

        # Build priors
        regr_prior = Distribution.load(learner_parameters['regr'])
        corr_prior = Distribution.load(learner_parameters['corr'])
        theta0_prior = Distribution.load(learner_parameters['theta0'])
        estimate_ml_prior = Distribution.load(learner_parameters['estimate_ml'], category_type='bool')
        thetaL_prior = Distribution.load(learner_parameters['thetaL'])
        thetaU_prior = Distribution.load(learner_parameters['thetaU'])
        normalize_prior = Distribution.load(learner_parameters['normalize'], category_type='bool')
        nugget_prior = Distribution.load(learner_parameters['nugget'])
        optimizer_prior = Distribution.load(learner_parameters['optimizer'])

        # Build parameters
        regr_param = Parameter('regr', prior=regr_prior, default='constant')
        corr_param = Parameter('corr', prior=corr_prior, default='squared_exponential')
        theta0_param = Parameter('theta0', prior=theta0_prior, default=0.1)
        estimate_ml_param = Parameter('estimate_ml', prior=estimate_ml_prior, default=False)
        thetaU_param = Parameter('thetaU', prior=thetaU_prior, default=None,
                                 sample_if=lambda params: params['estimate_ml'])
        thetaL_param = Parameter('thetaL', prior=thetaL_prior, default=None,
                                 sample_if=lambda params: params['estimate_ml'],
                                 valid_if=lambda value, params: params['thetaU'] > value,
                                 action_if_invalid='resample')
        normalize_param = Parameter('normalize', prior=normalize_prior, default=True)
        nugget_param = Parameter('nugget', prior=nugget_prior, default=10 * np.finfo(float).eps)
        optimizer_param = Parameter('optimizer', prior=optimizer_prior, default='fmin_cobyla')

        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['regr'] = regr_param
        param_space['corr'] = corr_param
        param_space['theta0'] = theta0_param
        param_space['estimate_ml'] = estimate_ml_param
        param_space['thetaL'] = thetaL_param
        param_space['thetaU'] = thetaU_param
        param_space['normalize'] = normalize_param
        param_space['nugget'] = nugget_param
        param_space['optimizer'] = optimizer_param
        param_space['random_state'] = random_state_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['GaussianProcessClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg(self):
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        # DOES NOT WORLK
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['regr'] = param.CategoricalParameter('regr',
                                                         ['constant', 'linear', 'quadratic'],
                                                         default='constant')
        # Generalized exponential correlation misbehaves. Check sklearn code
        #param_space['corr'] = param.CategoricalParameter('corr', ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear'], default='squared_exponential')
        param_space['corr'] = param.CategoricalParameter('corr', ['absolute_exponential', 'squared_exponential', 'cubic', 'linear'], default='squared_exponential')
        #param_space['corr'].categories['generalized_exponential']['p'] = param.NumericalParameter('p', prior=param.LogUniform(-5, 5), default=0)
        param_space['estimate_ml'] = param.CategoricalParameter('estimate_ml', {True: 0.5, False: 0.5}, default=False)
        param_space['estimate_ml'].categories[True]['thetaU'] = param.NumericalParameter('thetaU', prior=param.Uniform(), default=None, valid_if="params['thetaL'] < value", when_invalid='resample')
        param_space['estimate_ml'].categories[True]['thetaL'] = param.NumericalParameter('thetaL', prior=param.Uniform(), default=None)
        param_space['normalize'] = param.CategoricalParameter('normalize', [True, False], default=True)
        param_space['optimizer'] = param.CategoricalParameter('optimizer', ['fmin_cobyla', 'Welch'], default='fmin_cobyla')

        param_space['theta0'] = param.NumericalParameter('theta0', prior=param.Uniform(), default=0.1)
        param_space['nugget'] = param.NumericalParameter('nugget', prior=param.Uniform(lower=5 * np.finfo(float).eps, upper=1.0), default=10 * np.finfo(float).eps)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier (libsvm)."""

    def create_classifier(self, **parameters):
        """Creates the support vector machine classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Support Vector Machine classifier.
        """
        if 'kernel' in parameters:
            parameters['kernel'] = str(parameters['kernel'])  # in case numpy.string_ is passed
        parameters['probability'] = True  # To enable predict_proba
        classifier = SVC(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['C'] = ChoiceParameter([1.0])
        param_space['kernel'] = ChoiceParameter(['linear'])
        param_space['degree'] = ChoiceParameter([3])  # ?
        param_space['coef0'] = ChoiceParameter([0.0])
        param_space['gamma'] = ChoiceParameter([0.0])
        param_space['shrinking'] = BooleanParameter()
        param_space['tol'] = ChoiceParameter([1e-3])
        param_space['probability'] = ChoiceParameter([True])  # MUST BE SET TO TRUE!

        '''
        param_space['C'] = ChoiceParameter([1.0])
        param_space['kernel'] = ChoiceParameter(['linear', 'poly', 'rbf', 'sigmoid'])
        param_space['degree'] = ChoiceParameter([3])  # ?
        param_space['coef0'] = ChoiceParameter([0.0])
        param_space['gamma'] = ChoiceParameter([0.0])
        param_space['shrinking'] = BooleanParameter()
        param_space['tol'] = ChoiceParameter([1e-3])
        param_space['probability'] = ChoiceParameter([True])  # MUST BE SET TO TRUE!
        #param_space['class_weight'] = ?
        # param_space['random_state'] = ?
        '''
        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            C_options = OrderedDict()
            C_options['distribution'] = 'LogNormal'  # ?
            C_options['mean'] = 1.0
            C_options['stdev'] = 0.5

            kernel_options = OrderedDict()
            kernel_options['distribution'] = 'Categorical'
            kernel_options['categories'] = ['linear', 'poly', 'rbf', 'sigmoid']
            kernel_options['probabilities'] = [0.2, 0.2, 0.4, 0.2]

            degree_options = OrderedDict()
            degree_options['distribution'] = 'Uniform'
            degree_options['lower'] = 2
            degree_options['upper'] = 5
            #degree_options['discretize'] = 'round'

            gamma_options = OrderedDict()
            gamma_options['distribution'] = 'Uniform'
            gamma_options['lower'] = 0.0
            gamma_options['upper'] = 1.0

            coef0_options = OrderedDict()
            coef0_options['distribution'] = 'Uniform'
            coef0_options['lower'] = 0.0
            coef0_options['upper'] = 1.0

            shrinking_options = OrderedDict()
            shrinking_options['distribution'] = 'Categorical'
            shrinking_options['categories'] = [True, False]
            shrinking_options['probabilities'] = [0.75, 0.25]

            tolerance_options = OrderedDict()
            tolerance_options['distribution'] = 'LogNormal'
            tolerance_options['mean'] = 1e-4
            tolerance_options['stdev'] = 1.0

            class_weight_options = OrderedDict()
            class_weight_options['distribution'] = 'Categorical'
            class_weight_options['categories'] = ['auto', None]
            class_weight_options['probabilities'] = [0.75, 0.25]

            learner_options = {'C': C_options,
                               'kernel': kernel_options,
                               'degree': degree_options,
                               'gamma': gamma_options,
                               'coef0': coef0_options,
                               'shrinking': shrinking_options,
                               'tolerance': tolerance_options,
                               'class_weight': class_weight_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['SVMClassifier'][optimizer]

        # Build priors
        C_prior = Distribution.load(learner_parameters['C'])
        kernel_prior = Distribution.load(learner_parameters['kernel'])
        degree_prior = Distribution.load(learner_parameters['degree'])
        gamma_prior = Distribution.load(learner_parameters['gamma'])
        coef0_prior = Distribution.load(learner_parameters['coef0'])
        shrinking_prior = Distribution.load(learner_parameters['shrinking'], category_type='bool')
        tolerance_prior = Distribution.load(learner_parameters['tolerance'])
        class_weight_prior = Distribution.load(learner_parameters['class_weight'], parse_none=True)

        # Build parameters
        C_param = Parameter('C', prior=C_prior, default=1.0)
        kernel_param = Parameter('kernel', prior=kernel_prior, default='rbf')
        degree_param = Parameter('degree', prior=degree_prior, default=3,
                                 sample_if=lambda params: params['kernel'] == 'poly',
                                 discretize='round')
        gamma_param = Parameter('gamma', prior=gamma_prior, default=0.0,
                                sample_if=lambda params: params['kernel'] in ('rbf', 'poly', 'sigmoid', ))
        coef0_param = Parameter('coef0', prior=coef0_prior, default=0.0,
                                sample_if=lambda params: params['kernel'] in ('poly', 'sigmoid', ))
        shrinking_param = Parameter('shrinking', prior=shrinking_prior, default=True)
        tolerance_param = Parameter('tol', prior=tolerance_prior, default=1e-3)
        class_weight_param = Parameter('class_weight', prior=class_weight_prior, default=None)
        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['C'] = C_param
        param_space['kernel'] = kernel_param
        param_space['degree'] = degree_param
        param_space['gamma'] = gamma_param
        param_space['coef0'] = coef0_param
        param_space['shrinking'] = shrinking_param
        param_space['tol'] = tolerance_param
        param_space['class_weight'] = class_weight_param
        param_space['random_state'] = random_state_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['SVMClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['kernel'] = param.CategoricalParameter('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], default='rbf')
        param_degree = param.NumericalParameter('degree', prior=param.Uniform(2, 5), default=3, discretize=True)
        param_gamma = param.NumericalParameter('gamma', prior=param.Uniform(), default=0.0)
        param_coef0 = param.NumericalParameter('coef0', prior=param.Uniform(), default=0.0)
        param_space['kernel'].categories['poly']['degree'] = param_degree
        param_space['kernel'].categories['poly']['gamma'] = param_gamma
        param_space['kernel'].categories['poly']['coef0'] = param_coef0
        param_space['kernel'].categories['rbf']['gamma'] = param_gamma
        param_space['kernel'].categories['sigmoid']['gamma'] = param_gamma
        param_space['kernel'].categories['sigmoid']['coef0'] = param_coef0
        param_space['shrinking'] = param.CategoricalParameter('shrinking', {True: 0.5, False: 0.5}, default=False)
        param_space['class_weight'] = param.CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)

        # Numerical parameters
        param_space['C'] = param.NumericalParameter('C', prior=param.LogUniform(-10.0, 10.0), default=0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=-5, stdev=0.5), default=0.001)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Common numerical hyperparameters
        C = param.NumericalParameter('C', prior=param.LogUniform(-10.0, 10.0), default=0)
        tol = param.NumericalParameter('tol', prior=param.LogNormal(mean=-5, stdev=0.5), default=0.001)

        kernel_cats = ['linear', 'poly', 'rbf', 'sigmoid']
        param_space['kernel'] = param.CategoricalParameter('kernel', kernel_cats, default='rbf')
        for kernel in kernel_cats:
            shrinking_cats = [True, False]
            kernel_branch = param_space['kernel'].categories[kernel]
            kernel_branch['shrinking'] = param.CategoricalParameter('shrinking', shrinking_cats, default=False)
            for shrinking in shrinking_cats:
                class_weight_cats = [None, 'auto']
                shrinking_branch = kernel_branch['shrinking'].categories[shrinking]
                shrinking_branch['class_weight'] = param.CategoricalParameter('class_weight', class_weight_cats, default=None)
                for class_weight in class_weight_cats:  # Maybe this is not needed
                    class_weight_branch = shrinking_branch['class_weight'].categories[class_weight]
                    class_weight_branch['C'] = C
                    class_weight_branch['tol'] = tol

        param_degree = param.NumericalParameter('degree', prior=param.Uniform(2, 5), default=3, discretize=True)
        param_gamma = param.NumericalParameter('gamma', prior=param.Uniform(), default=0.0)
        param_coef0 = param.NumericalParameter('coef0', prior=param.Uniform(), default=0.0)
        param_space['kernel'].categories['poly']['degree'] = param_degree
        param_space['kernel'].categories['poly']['gamma'] = param_gamma
        param_space['kernel'].categories['poly']['coef0'] = param_coef0
        param_space['kernel'].categories['rbf']['gamma'] = param_gamma
        param_space['kernel'].categories['sigmoid']['gamma'] = param_gamma
        param_space['kernel'].categories['sigmoid']['coef0'] = param_coef0

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class LinearSVMClassifier(BaseClassifier):
    """Linear Support Vector Machine classifier implementation (liblinear)."""
    def create_classifier(self, **parameters):
        """Creates the linear Support Vector Machine classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Linear Support Vector Machine classifier.
        """
        if 'ovr_valid' in parameters:
            penalty, loss, dual = parameters['ovr_valid'].split('_')
            parameters['loss'] = loss
            parameters['penalty'] = penalty
            parameters['dual'] = dual.lower() == 'true'
            del parameters['ovr_valid']
        classifier = LinearSVC(**parameters)
        #Log.write("{classifier}: {params}".format(classifier=self.__class__.__name__, params=parameters))
        return classifier

    #@log_step("Predicting with Linear SVM Classifier")
    def predict(self, data):
        """Perform the prediction step on the given data.

        :param data: numpy array with data to predict.
        """
        self.predict_data = data
        linear_svc = self.classifier
        # LinearSVC doesn't implement a predict_proba method
        prediction = linear_svc.predict(data)

        prediction = array_to_proba(prediction, min_columns=len(self.training_classes))
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()
        param_space['dual'] = BooleanParameter()
        param_space['tol'] = ChoiceParameter([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3])
        #param_space['tol'] = ChoiceParameter([1e-4, 1e-3])
        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            C_options = OrderedDict()
            C_options['distribution'] = 'LogNormal'  # ?
            C_options['mean'] = 1.0
            C_options['stdev'] = 0.5

            ovr_valid_options = OrderedDict()  # Check sklearn/svm/base.py _get_solver_type
            ovr_valid_options['distribution'] = 'Categorical'
            ovr_valid_options['categories'] = ['l1_l2_False', 'l2_l1_True', 'l2_l2_True', 'l2_l2_False']

            '''
            loss_options = OrderedDict()
            loss_options['distribution'] = 'Categorical'
            loss_options['categories'] = ['l1', 'l2']

            penalty_options = OrderedDict()
            penalty_options['distribution'] = 'Categorical'
            penalty_options['categories'] = ['l1', 'l2']

            dual_options = OrderedDict()
            dual_options['distribution'] = 'Categorical'
            dual_options['categories'] = [True, False]
            dual_options['probabilities'] = [0.5, 0.5]
            '''

            tolerance_options = OrderedDict()
            tolerance_options['distribution'] = 'LogNormal'
            tolerance_options['mean'] = 1e-4
            tolerance_options['stdev'] = 1.0

            multi_class_options = OrderedDict()
            multi_class_options['distribution'] = 'Categorical'
            multi_class_options['categories'] = ['ovr', 'crammer_singer']
            multi_class_options['probabilities'] = [0.75, 0.25]

            fit_intercept_options = OrderedDict()
            fit_intercept_options['distribution'] = 'Categorical'
            fit_intercept_options['categories'] = [True, False]
            fit_intercept_options['probabilities'] = [0.5, 0.5]

            intercept_scaling_options = OrderedDict()
            intercept_scaling_options['distribution'] = 'Uniform'
            intercept_scaling_options['lower'] = 0.0
            intercept_scaling_options['upper'] = 5.0

            class_weight_options = OrderedDict()
            class_weight_options['distribution'] = 'Categorical'
            class_weight_options['categories'] = [None, 'auto']
            class_weight_options['probabilities'] = [0.5, 0.5]

            learner_options = {'C': C_options,
                               'ovr_valid': ovr_valid_options,
                               #'loss': loss_options,
                               #'penalty': penalty_options,
                               #'dual': dual_options,
                               'tolerance': tolerance_options,
                               'multi_class': multi_class_options,
                               'fit_intercept': fit_intercept_options,
                               'intercept_scaling': intercept_scaling_options,
                               'class_weight': class_weight_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['LinearSVMClassifier'][optimizer]

        # Build priors
        C_prior = Distribution.load(learner_parameters['C'])
        ovr_valid_prior = Distribution.load(learner_parameters['ovr_valid'])
        """
        loss_prior = Distribution.load(learner_parameters['loss'])
        penalty_prior = Distribution.load(learner_parameters['penalty'])
        dual_prior = Distribution.load(learner_parameters['dual'], category_type='bool')
        """
        tolerance_prior = Distribution.load(learner_parameters['tolerance'])
        multi_class_prior = Distribution.load(learner_parameters['multi_class'])
        fit_intercept_prior = Distribution.load(learner_parameters['fit_intercept'], category_type='bool')
        intercept_scaling_prior = Distribution.load(learner_parameters['intercept_scaling'])
        class_weight_prior = Distribution.load(learner_parameters['class_weight'], parse_none=True)

        # Build parameters
        C_param = Parameter('C', prior=C_prior, default=1.0)
        ovr_valid_param = Parameter('ovr_valid', prior=ovr_valid_prior, default='l2_l2_True',
                                    sample_if=lambda params: params['multi_class'] != 'crammer_singer')
        """
        loss_param = Parameter('loss', prior=loss_prior, default='l2',
                               sample_if=lambda params: params['multi_class'] != 'crammer_singer')
        penalty_param = Parameter('penalty', prior=penalty_prior, default='l2',
                                  sample_if=lambda params: params['multi_class'] != 'crammer_singer')
        dual_param = Parameter('dual', prior=dual_prior, default=True,
                               sample_if=lambda params: params['multi_class'] != 'crammer_singer')
        """
        tolerance_param = Parameter('tolerance', prior=tolerance_prior, default=1e-4)
        multi_class_param = Parameter('multi_class', prior=multi_class_prior, default='ovr')
        fit_intercept_param = Parameter('fit_intercept', prior=fit_intercept_prior, default=True)
        intercept_scaling_param = Parameter('intercept_scaling', prior=intercept_scaling_prior, default=1,
                                            sample_if=lambda params: params['fit_intercept'])

        class_weight_param = Parameter('class_weight', prior=class_weight_prior, default=None)
        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['C'] = C_param
        param_space['ovr_valid'] = ovr_valid_param
        """
        param_space['loss'] = loss_param
        param_space['penalty'] = penalty_param
        param_space['dual'] = dual_param
        """
        param_space['tol'] = tolerance_param
        param_space['multi_class'] = multi_class_param
        param_space['fit_intercept'] = fit_intercept_param
        param_space['intercept_scaling'] = intercept_scaling_param
        param_space['class_weight'] = class_weight_param
        param_space['random_state'] = random_state_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['LinearSVMClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['multi_class'] = param.CategoricalParameter('multi_class', ['ovr', 'crammer_singer'], default='ovr')
        param_space['multi_class'].categories['ovr']['ovr_valid'] = param.CategoricalParameter('ovr_valid', ['l1_l2_False', 'l2_l1_True', 'l2_l2_True', 'l2_l2_False'], default='l2_l2_True')
        param_space['fit_intercept'] = param.CategoricalParameter('fit_intercept', {True: 0.5, False: 0.5}, default=False)
        param_space['fit_intercept'].categories[True]['intercept_scaling'] = param.NumericalParameter('intercept_scaling', prior=param.LogUniform(-10, 10), default=1.0)
        param_space['class_weight'] = param.CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)

        # Numerical parameters
        param_space['C'] = param.NumericalParameter('C', prior=param.LogUniform(-10.0, 10.0), default=1.0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=-5, stdev=0.5), default=0.0001)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()
        # Numerical parameters
        param_space['C'] = param.NumericalParameter('C', prior=param.LogUniform(-10.0, 10.0), default=1.0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=-5, stdev=0.5), default=0.0001)

        multi_class_cats = ['ovr', 'crammer_singer']
        param_space['multi_class'] = param.CategoricalParameter('multi_class', multi_class_cats, default='ovr')
        ovr_branch = param_space['multi_class'].categories['ovr']
        ovr_valid_cats = ['l1_l2_False', 'l2_l1_True', 'l2_l2_True', 'l2_l2_False']
        ovr_branch['ovr_valid'] = param.CategoricalParameter('ovr_valid', ovr_valid_cats, default='l2_l2_True')
        for ovr_valid in ovr_valid_cats:
            fit_intercept_cats = [True, False]
            ovr_valid_branch = ovr_branch['ovr_valid'].categories[ovr_valid]
            ovr_valid_branch['fit_intercept'] = param.CategoricalParameter('fit_intercept', fit_intercept_cats, default=False)
            for fit_intercept in fit_intercept_cats:
                fit_intercept_branch = ovr_valid_branch['fit_intercept'].categories[fit_intercept]
                fit_intercept_branch['class_weight'] = param.CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)

        crammer_singer_branch = param_space['multi_class'].categories['crammer_singer']
        fit_intercept_cats = [True, False]
        crammer_singer_branch['fit_intercept'] = param.CategoricalParameter('fit_intercept', fit_intercept_cats, default=False)
        for fit_intercept in fit_intercept_cats:
            fit_intercept_branch = crammer_singer_branch['fit_intercept'].categories[fit_intercept]
            fit_intercept_branch['class_weight'] = param.CategoricalParameter('class_weight', {None: 0.5, 'auto': 0.5}, default=None)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class NuSVMClassifier(BaseClassifier):
    """Nu-Support Vector Machine classifier."""

    def create_classifier(self, **parameters):
        """Creates the Nu-support vector classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Nu-support Vector classifier
        """
        if 'kernel' in parameters:
            parameters['kernel'] = str(parameters['kernel'])  # in case numpy.string_ is passed
        parameters['probability'] = True  # To enable predict_proba
        classifier = NuSVC(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['nu'] = ChoiceParameter([0.5])
        param_space['kernel'] = ChoiceParameter(['linear', 'poly', 'rbf', 'sigmoid'])
        #                                         'precomputed'])
        param_space['degree'] = ChoiceParameter([3])  # ?
        param_space['coef0'] = ChoiceParameter([0.0])
        param_space['gamma'] = ChoiceParameter([0.0])
        param_space['shrinking'] = BooleanParameter()
        param_space['tol'] = ChoiceParameter([1e-3])
        param_space['probability'] = ChoiceParameter([True])  # MUST BE SET TO TRUE!
        # param_space['class_weight'] = ?
        # param_space['random_state'] = ?

        return param_space

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            nu_options = OrderedDict()
            nu_options['distribution'] = 'Uniform'
            nu_options['lower'] = 0.0
            nu_options['upper'] = 1.0

            kernel_options = OrderedDict()
            kernel_options['distribution'] = 'Categorical'
            kernel_options['categories'] = ['linear', 'poly', 'rbf', 'sigmoid']
            kernel_options['probabilities'] = [0.2, 0.2, 0.4, 0.2]

            degree_options = OrderedDict()
            degree_options['distribution'] = 'Uniform'
            degree_options['lower'] = 2
            degree_options['upper'] = 5

            gamma_options = OrderedDict()
            gamma_options['distribution'] = 'Uniform'
            gamma_options['lower'] = 0.0
            gamma_options['upper'] = 1.0

            coef0_options = OrderedDict()
            coef0_options['distribution'] = 'Uniform'
            coef0_options['lower'] = 0.0
            coef0_options['upper'] = 1.0

            shrinking_options = OrderedDict()
            shrinking_options['distribution'] = 'Categorical'
            shrinking_options['categories'] = [True, False]
            shrinking_options['probabilities'] = [0.75, 0.25]

            tolerance_options = OrderedDict()
            tolerance_options['distribution'] = 'LogNormal'
            tolerance_options['mean'] = 1e-4
            tolerance_options['stdev'] = 1.0

            learner_options = {'nu': nu_options,
                               'kernel': kernel_options,
                               'degree': degree_options,
                               'gamma': gamma_options,
                               'coef0': coef0_options,
                               'shrinking': shrinking_options,
                               'tolerance': tolerance_options,
                               'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['NuSVMClassifier'][optimizer]

        # Build priors
        nu_prior = Distribution.load(learner_parameters['nu'])
        kernel_prior = Distribution.load(learner_parameters['kernel'])
        degree_prior = Distribution.load(learner_parameters['degree'])
        gamma_prior = Distribution.load(learner_parameters['gamma'])
        coef0_prior = Distribution.load(learner_parameters['coef0'])
        shrinking_prior = Distribution.load(learner_parameters['shrinking'], category_type='bool')
        tolerance_prior = Distribution.load(learner_parameters['tolerance'])

        # Build parameters
        nu_param = Parameter('nu', prior=nu_prior, default=0.5)
        kernel_param = Parameter('kernel', prior=kernel_prior, default='rbf')
        degree_param = Parameter('degree', prior=degree_prior, default=3,
                                 sample_if=lambda params: params['kernel'] == 'poly',
                                 discretize='round')
        gamma_param = Parameter('gamma', prior=gamma_prior, default=0.0,
                                sample_if=lambda params: params['kernel'] in ('rbf', 'poly', 'sigmoid', ))
        coef0_param = Parameter('coef0', prior=coef0_prior, default=0.0,
                                sample_if=lambda params: params['kernel'] in ('poly', 'sigmoid', ))
        shrinking_param = Parameter('shrinking', prior=shrinking_prior, default=True)
        tolerance_param = Parameter('tol', prior=tolerance_prior, default=1e-3)
        random_state_param = parameters['Global'].as_int('randomstate')

        # Create parameter space
        param_space = ParameterSpace()
        param_space['nu'] = nu_param
        param_space['kernel'] = kernel_param
        param_space['degree'] = degree_param
        param_space['gamma'] = gamma_param
        param_space['coef0'] = coef0_param
        param_space['shrinking'] = shrinking_param
        param_space['tol'] = tolerance_param
        param_space['random_state'] = random_state_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['NuSVMClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        '''
        param_space['kernel'] = param.CategoricalParameter('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], default='rbf')
        param_degree = param.NumericalParameter('degree', prior=param.Uniform(2, 5), default=3, discretize=True)
        param_gamma = param.NumericalParameter('gamma', prior=param.Uniform(), default=0.0)
        param_coef0 = param.NumericalParameter('coef0', prior=param.Uniform(), default=0.0)
        param_space['kernel'].categories['poly']['degree'] = param_degree
        param_space['kernel'].categories['poly']['gamma'] = param_gamma
        param_space['kernel'].categories['poly']['coef0'] = param_coef0
        param_space['kernel'].categories['rbf']['gamma'] = param_gamma
        param_space['kernel'].categories['sigmoid']['gamma'] = param_gamma
        param_space['kernel'].categories['sigmoid']['coef0'] = param_coef0
        param_space['shrinking'] = param.CategoricalParameter('shrinking', {True: 0.5, False: 0.5}, default=False)
        '''

        # Numerical parameters
        param_space['nu'] = param.NumericalParameter('nu', prior=param.Uniform(), default=0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=-5, stdev=0.5), default=0.5)

        shrinking_cats = [True, False]
        shrinking_param = param.CategoricalParameter('shrinking', shrinking_cats, default=False)

        kernel_cats = ['linear', 'poly', 'rbf', 'sigmoid']
        kernel_param = param.CategoricalParameter('kernel',  kernel_cats, default='rbf')
        for kernel in kernel_cats:
            kernel_branch = kernel_param.categories[kernel]
            kernel_branch['shrinking'] = shrinking_param

        param_degree = param.NumericalParameter('degree', prior=param.Uniform(2, 5), default=3, discretize=True)
        param_gamma = param.NumericalParameter('gamma', prior=param.Uniform(), default=0.0)
        param_coef0 = param.NumericalParameter('coef0', prior=param.Uniform(), default=0.0)
        kernel_param.categories['poly']['degree'] = param_degree
        kernel_param.categories['poly']['gamma'] = param_gamma
        kernel_param.categories['poly']['coef0'] = param_coef0

        #param_space['kernel'].categories['poly']['shrinking'] = shrinking_param
        #param_space['kernel'].categories['poly']['shrinking'] = shrinking_param
        #param_space['kernel'].categories['poly']['shrinking'] = shrinking_param
        #param_space['kernel'].categories['poly']['shrinking'] = shrinking_param

        kernel_param.categories['rbf']['gamma'] = param_gamma
        kernel_param.categories['sigmoid']['gamma'] = param_gamma
        kernel_param.categories['sigmoid']['coef0'] = param_coef0
        param_space['kernel'] = kernel_param

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options

    @classmethod
    def get_default_cfg2(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        param_space['kernel'] = param.CategoricalParameter('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], default='rbf')
        param_degree = param.NumericalParameter('degree', prior=param.Uniform(2, 5), default=3, discretize=True)
        param_gamma = param.NumericalParameter('gamma', prior=param.Uniform(), default=0.0)
        param_coef0 = param.NumericalParameter('coef0', prior=param.Uniform(), default=0.0)
        param_space['kernel'].categories['poly']['degree'] = param_degree
        param_space['kernel'].categories['poly']['gamma'] = param_gamma
        param_space['kernel'].categories['poly']['coef0'] = param_coef0
        param_space['kernel'].categories['rbf']['gamma'] = param_gamma
        param_space['kernel'].categories['sigmoid']['gamma'] = param_gamma
        param_space['kernel'].categories['sigmoid']['coef0'] = param_coef0
        param_space['shrinking'] = param.CategoricalParameter('shrinking', {True: 0.5, False: 0.5}, default=False)

        # Numerical parameters
        param_space['nu'] = param.NumericalParameter('nu', prior=param.Uniform(), default=0)
        param_space['tol'] = param.NumericalParameter('tol', prior=param.LogNormal(mean=-5, stdev=0.5), default=0.5)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class LinearDiscriminantClassifier(BaseClassifier):
    """Linear Discriminant classifier."""

    def create_classifier(self, **parameters):
        """Creates the linear discriminant classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Linear Discriminant classifier
        """
        classifier = LDA(**parameters)
        return classifier

    def train(self, data, target):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(LinearDiscriminantClassifier, self).train(data, target)
            return trained

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            learner_options = {'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        #learner_parameters = parameters['Classifiers']['LinearDiscriminantClassifier'][optimizer]

        # Create parameter space
        param_space = ParameterSpace()
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['LinearDiscriminantClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options


class QuadraticDiscriminantClassifier(BaseClassifier):
    """Quadratic discriminant classifier."""

    def train(self, data, target):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(QuadraticDiscriminantClassifier, self).train(data, target)
            return trained

    def create_classifier(self, **parameters):
        """Creates the quadratic discriminant classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Quadratic Discriminant classifier
        """
        classifier = QDA(**parameters)
        return classifier

    @classmethod
    def get_default_config(self, optimizer=None):
        if optimizer == 'KDEOptimizer':
            reg_param_options = OrderedDict()
            reg_param_options['distribution'] = 'Uniform'
            reg_param_options['lower'] = 0.0
            reg_param_options['upper'] = 1.0

            learner_options = {'reg_param': reg_param_options, 'enabled': True}
        elif optimizer is None:
            learner_options = OrderedDict()
            for optimizer in ('KDEOptimizer',):
                learner_options[optimizer] = self.get_default_config(optimizer)
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters, optimizer):
        # Read learner settings to build priors
        if self.__name__ not in parameters['Classifiers']:
            parameters['Classifiers'][self.__name__] = self.get_default_config()
        learner_parameters = parameters['Classifiers'][self.__name__][optimizer]

        # Build priors
        reg_param_prior = Distribution.load(learner_parameters['reg_param'])

        # Build parameters
        reg_param = Parameter('reg_param', prior=reg_param_prior, default=0.0)

        # Create parameter space
        param_space = ParameterSpace()
        param_space['reg_param'] = reg_param
        return param_space

    @classmethod
    def create_param_space(self, parameters):
        from ..parameters import param
        learner_parameters = parameters['Classifiers']['QuadraticDiscriminantClassifier']
        param_space = param.ParameterSpace.load(learner_parameters)
        return param_space

    @classmethod
    def get_default_cfg(self):
        from ..parameters import param
        param_space = param.ParameterSpace()

        # Numerical parameters
        param_space['reg_param'] = param.NumericalParameter('reg_param', prior=param.Uniform(), default=0.0)

        # return learner_options
        learner_options = param_space.dump()
        learner_options['enabled'] = '$True'
        return learner_options
