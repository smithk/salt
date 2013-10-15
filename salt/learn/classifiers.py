"""The :mod:`salt.learning.trainers` module implements classes for learning."""

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from ..utils.debug import log_step, Log
from .base import BaseLearner
from ..parameters.base import (ParameterSpace, ChoiceParameter,
                               BooleanParameter, )


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
        raise NotImplementedError((self, dataset))

    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        raise NotImplementedError((self, dataset))

    def get_learner_parameters(self, parameters):
        """Convert parameter space into parameters expected by the learner."""
        return (), {}

    @classmethod
    def create_default_params(self):
        raise NotImplementedError


class LinearSVMClassifier(BaseClassifier):
    """Linear Support Vector Machine classifier implementation."""
    def create_classifier(self, **parameters):
        """Creates the linear Support Vector Machine classifier class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Linear Support Vector Machine classifier.
        """
        classifier = LinearSVC(**parameters)
        #Log.write("{classifier}: {params}".format(classifier=self.__class__.__name__, params=parameters))
        return classifier

    #@log_step("Training with Linear SVM Classifier")
    def train(self, dataset):
        """Perform the training step on the given dataset.

        :param dataset: Training dataset.
        """
        self.training_dataset = dataset
        linear_svc = self.classifier
        trained = linear_svc.fit(dataset.data, dataset.target)
        return trained

    #@log_step("Predicting with Linear SVM Classifier")
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        linear_svc = self.classifier
        prediction = linear_svc.predict(dataset.data)
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()
        param_space['dual'] = BooleanParameter()
        param_space['tol'] = ChoiceParameter([1e-4, 2e-4, 3e-4, 4e-4,5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3])
        #param_space['tol'] = ChoiceParameter([1e-4, 1e-3])
        return param_space


class GaussianNaiveBayesClassifier(BaseClassifier):
    """Gaussian Naive Bayes classifier implementation."""
    def create_classifier(self, **parameters):
        """Creates the linear Gaussian Naive Bayes classifier class, with the options given.

        :param parameters: parameters to pass to the constructor.
        :returns: Gaussian Naive Bayes classifier.
        """
        classifier = GaussianNB()
        #Log.write("{classifier}: {params}".format(classifier=self.__class__.__name__, params=parameters))
        return classifier

    #@log_step("Training with Gaussian Naive Bayes classifier")
    def train(self, dataset):
        """Perform the training step on the given dataset.

        :param dataset: Training dataset.
        """
        self.training_dataset = dataset
        gaussian_naive_bayes = self.classifier
        trained = gaussian_naive_bayes.fit(dataset.data, dataset.target)
        return trained

    #@log_step("Predicting with Gaussian Naive Bayes classifier")
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        gaussian_naive_bayes = self.classifier
        prediction = gaussian_naive_bayes.predict(dataset.data)
        return prediction

    @classmethod
    def create_default_params(self):
        return ParameterSpace()


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

    #@log_step("Training with Knn classifier")
    def train(self, dataset):
        """Perform the training step on the given dataset.

        :param dataset: Training dataset.
        """
        self.training_dataset = dataset
        knn = self.classifier
        trained = knn.fit(dataset.data, dataset.target)
        return trained

    #@log_step("Predicting with Knn classifier")
    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        knn = self.classifier
        prediction = knn.predict(dataset.data)
        return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()
        #"""
        param_space['n_neighbors'] = ChoiceParameter([2, 3, 4, 5, 6, 7, 8, 9, 10])
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
