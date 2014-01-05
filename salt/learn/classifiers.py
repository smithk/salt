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
                              # AdaBoostClassifier as SKAdaBoostClassifier,
                              GradientBoostingClassifier as SKGradientBoostingClassifier)
from sklearn.gaussian_process import GaussianProcess
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.lda import LDA
from sklearn.qda import QDA
import numpy as np
#from ..utils.debug import log_step  # , Log
from ..utils.arrays import array_to_proba
from .base import BaseLearner
from ..parameters import (ParameterSpace, ChoiceParameter,
                               BooleanParameter, )
from ..parameters import Distribution, LogUniformDist, UniformDist, LogNormalDist, Parameter
from collections import OrderedDict


class BaseClassifier(BaseLearner):
    """Base class for classifiers. All classifiers should inherit from this class."""
    def __init__(self, **parameters):
        self._parameters = None
        self._classifier = None
        self.training_dataset = None
        self.predict_dataset = None
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

    #pylint: disable=W0212
    parameters = property(lambda self: self._parameters, _update_parameters)
    classifier = property(lambda self: self._get_classifier())

    # --- Public methods ---

    def create_classifier(self, **parameters):
        """Call the classifier constructor for the specific class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        """
        raise NotImplementedError((self, parameters))

    def train(self, dataset):
        """Perform the training step on the given dataset.

        :param dataset: Training dataset.
        """
        self.training_dataset = dataset
        classifier = self.classifier
        self.training_classes = np.unique(dataset.target)
        trained = classifier.fit(dataset.data, dataset.target.astype(int))
        return trained

    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        classifier = self.classifier
        prediction = classifier.predict_proba(dataset.data)
        # extended_prediction includes columns for non-observed classes
        extended_prediction = np.zeros((len(dataset.data), len(dataset.target_names)))
        prediction_index = 0
        for i in np.unique(self.training_classes):
            extended_prediction[:, i] = prediction[:, prediction_index]
            prediction_index += 1
        return extended_prediction

    def get_learner_parameters(self, parameters):
        # DEPRECATED
        """Convert parameter space into parameters expected by the learner."""
        return (), {}

    def create_parameter_space(self, parameters=None):
        return (), {}  # args, kwargs

    @classmethod
    def create_default_params(self):
        raise NotImplementedError


class BaselineClassifier(BaseClassifier):
    def train(self, dataset):
        self._weights = np.bincount(dataset.target.astype(int),
                                    minlength=len(dataset.target_names)) / (1.0 * len(dataset.target))

    def predict(self, dataset):
        return np.repeat([self._weights], len(dataset.data), axis=0)


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
    def get_default_config(self):
        penalty_options = OrderedDict()
        penalty_options["distribution"] = "Categorical"
        penalty_options["categories"] = ['l1', 'l2']
        penalty_options["probabilities"] = [0.05, 0.95]

        dual_options = OrderedDict()
        dual_options["distribution"] = "Categorical"
        dual_options["categories"] = [True, False]
        dual_options["probabilities"] = [0.25, 0.75]

        C_options = OrderedDict()
        C_options["distribution"] = "LogUniform"
        C_options["lower"] = 0
        C_options["upper"] = 2 ** 10

        fit_intercept_options = OrderedDict()
        fit_intercept_options["distribution"] = "Categorical"
        fit_intercept_options["categories"] = [True, False]
        fit_intercept_options["probabilities"] = [0.5, 0.5]

        intercept_scaling_options = OrderedDict()
        intercept_scaling_options["distribution"] = "Uniform"
        intercept_scaling_options["lower"] = 0.0
        intercept_scaling_options["upper"] = 5.0

        class_weight_options = OrderedDict()
        class_weight_options["distribution"] = "Categorical"
        class_weight_options["categories"] = [None, 'auto']
        class_weight_options["probabilities"] = [0.5, 0.5]

        tolerance_options = OrderedDict()
        tolerance_options["distribution"] = "LogNormal"
        tolerance_options["mean"] = 1e-4
        tolerance_options["stdev"] = 1.0

        learner_options = {'penalty': penalty_options,
                           'dual': dual_options,
                           'C': C_options,
                           'fit_intercept': fit_intercept_options,
                           'intercept_scaling': intercept_scaling_options,
                           'class_weight': class_weight_options,
                           'tolerance': tolerance_options}
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['LogisticRegressionClassifier']  # learner_settings['log_reg_classif']

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
    def get_default_config(self):
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
                           'warm_start': warm_start_options}
        return learner_options

    @classmethod
    def create_parameter_space(self, parameters):
        # Read learner settings to build priors
        learner_parameters = parameters['Classifiers']['SGDClassifier']

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

    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        if self.classifier.loss in ('log', 'modified_huber'):
            prediction = super(SGDClassifier, self).predict(dataset)
        else:
            self.predict_dataset = dataset
            sgd = self.classifier
            prediction = sgd.predict(dataset.data)
            prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
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

    #@log_step("Predicting with Passive-aggressive classifier")
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        pac = self.classifier
        prediction = pac.predict(dataset.data)
        prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['C'] = ChoiceParameter([1.0])
        param_space['fit_intercept'] = BooleanParameter()
        param_space['n_iter'] = ChoiceParameter([5])
        param_space['shuffle'] = BooleanParameter()
        # param_space['random_state'] = ?
        param_space['loss'] = ChoiceParameter(['hinge', 'squared_hinge'])

        return param_space

    def create_parameter_space(self, parameters):
        # Read learner parameters to build priors
        learner_parameters = parameters.learner_settings['sgd_classif']

        # Build priors

        # Build parameters

        # Create parameter space
        param_space = ParameterSpace()
        return param_space


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
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        ridge = self.classifier
        prediction = ridge.predict(dataset.data)
        prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
        return prediction

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
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        ridgecv = self.classifier
        prediction = ridgecv.predict(dataset.data)
        prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
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
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        gaussian_naive_bayes = self.classifier
        prediction = gaussian_naive_bayes.predict_proba(dataset.data)
        return prediction

    @classmethod
    def create_default_params(self):
        return ParameterSpace()


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
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        knn = self.classifier
        #prediction = knn.predict_proba(dataset.data)
        prediction = knn.predict(dataset.data)
        prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
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
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        classifier = self.classifier
        prediction = classifier.predict(dataset.data)
        prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
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
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        classifier = self.classifier
        prediction = classifier.predict(dataset.data)
        prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
        return prediction

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


class DecisionTreeClassifier(BaseClassifier):
    """Decision tree classifier."""

    def create_classifier(self, **parameters):
        """Creates the decision tree classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Decision tree classifier.
        """
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


class ExtraTreeClassifier(BaseClassifier):
    """Extremely randomized tree classifier."""

    def create_classifier(self, **parameters):
        """Creates the extremely randomized tree classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Extremely randomized decision tree classifier.
        """
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


class ExtraTreeEnsembleClassifier(BaseClassifier):
    """Classifier based on a number of randomized decision trees."""

    def create_classifier(self, **parameters):
        """Creates the extra trees class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Extra trees classifier.
        """
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


class GradientBoostingClassifier(BaseClassifier):
    """Gradient boosting classifier."""

    def create_classifier(self, **parameters):
        """Creates the gradient boosting class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Gradient boosting classifier.
        """
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


class GaussianProcessClassifier(BaseClassifier):
    """Gaussian Process classifier."""

    def create_classifier(self, **parameters):
        """Creates the Gaussian process classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Gaussian process classifier.
        """
        classifier = GaussianProcess(**parameters)
        return classifier

    #@log_step("Predicting with Gaussian process classifier")
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        classifier = self.classifier
        prediction = classifier.predict(dataset.data)
        prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
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


class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier (libsvm)."""

    def create_classifier(self, **parameters):
        """Creates the support vector machine classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Support Vector Machine classifier.
        """
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


class LinearSVMClassifier(BaseClassifier):
    """Linear Support Vector Machine classifier implementation (liblinear)."""
    def create_classifier(self, **parameters):
        """Creates the linear Support Vector Machine classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Linear Support Vector Machine classifier.
        """
        classifier = LinearSVC(**parameters)
        #Log.write("{classifier}: {params}".format(classifier=self.__class__.__name__, params=parameters))
        return classifier

    #@log_step("Predicting with Linear SVM Classifier")
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        linear_svc = self.classifier
        # LinearSVC doesn't implement a predict_proba method
        prediction = linear_svc.predict(dataset.data)

        prediction = array_to_proba(prediction, min_columns=len(dataset.target_names))
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()
        param_space['dual'] = BooleanParameter()
        param_space['tol'] = ChoiceParameter([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3])
        #param_space['tol'] = ChoiceParameter([1e-4, 1e-3])
        return param_space


class NuSVMClassifier(BaseClassifier):
    """Nu-Support Vector Machine classifier."""

    def create_classifier(self, **parameters):
        """Creates the Nu-support vector classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Nu-support Vector classifier
        """
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


class LinearDiscriminantClassifier(BaseClassifier):
    """Linear Discriminant classifier."""

    def create_classifier(self, **parameters):
        """Creates the linear discriminant classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Linear Discriminant classifier
        """
        classifier = LDA(**parameters)
        return classifier

    def train(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(LinearDiscriminantClassifier, self).train(dataset)
            return trained

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_components'] = ChoiceParameter([None])  # ?
        param_space['priors'] = ChoiceParameter([None])  # ?

        return param_space


class QuadraticDiscriminantClassifier(BaseClassifier):
    """Quadratic discriminant classifier."""

    def train(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(QuadraticDiscriminantClassifier, self).train(dataset)
            return trained

    def create_classifier(self, **parameters):
        """Creates the quadratic discriminant classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Quadratic Discriminant classifier
        """
        classifier = QDA(**parameters)
        return classifier

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['priors'] = ChoiceParameter([None])  # ?
        param_space['reg_param'] = ChoiceParameter([0.0])  # ?

        return param_space
