import numpy as np
import warnings
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, Lasso, LassoCV,
                                  LassoLars, LassoLarsCV, Lars, LarsCV, ElasticNet, ElasticNetCV,
                                  OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
                                  SGDRegressor as SKSGDRegressor,
                                  PassiveAggressiveRegressor as SKPassiveAggressiveRegressor)
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import (KNeighborsRegressor,
                               RadiusNeighborsRegressor as SKRadiusNeighborsRegressor)
from sklearn.gaussian_process import GaussianProcess
from sklearn.tree import DecisionTreeRegressor as SKDesicionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor as SKRandomForestRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor as SKGradientBoostingRegressor)
from .base import BaseLearner
from ..parameters import ParameterSpace, ChoiceParameter, BooleanParameter


class BaseRegressor(BaseLearner):
    """Base class for regressors. All regressors should inherit from this class."""
    def __init__(self, **parameters):
        self._parameters = None
        self._regressor = None
        self.training_dataset = None
        self.predict_dataset = None
        self._update_parameters(parameters)

    def _get_regressor(self):
        """Get a reference to a valid regressor object."""
        if not self._regressor:  # Ensures to provide a reference to a trainer
            self._regressor = self.create_regressor(**self.parameters)
        return self._regressor

    def _update_parameters(self, parameters):
        """Update learner parameters."""
        self._regressor = None  # forces to re-train on next use
        self._parameters = parameters

    #pylint: disable=W0212
    parameters = property(lambda self: self._parameters, _update_parameters)
    regressor = property(lambda self: self._get_regressor())

    # --- Public methods ---

    def create_regressor(self, **parameters):
        """Call the regressor constructor for the specific class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        """
        raise NotImplementedError((self, parameters))

    def train(self, dataset):
        """Perform the training step on the given dataset.

        :param dataset: Training dataset.
        """
        self.training_dataset = dataset
        regressor = self.regressor
        trained = regressor.fit(dataset.data, dataset.target)
        return trained

    def predict(self, dataset):
        """Perform the prediction step on the given dataset.

        :param dataset: Dataset to predict.
        """
        self.predict_dataset = dataset
        regressor = self.regressor
        prediction = regressor.predict(dataset.data)
        return prediction

    def get_learner_parameters(self, parameters):
        """Convert parameter space into parameters expected by the learner."""
        return (), {}

    @classmethod
    def create_default_params(self):
        raise NotImplementedError


class BaselineRegressor(BaseRegressor):
    def __init__(self, **parameter):
        super(BaselineRegressor, self).__init__(**parameter)
        self._mean = None
        self._std = None
        self._q = None

    def train(self, dataset):
        self._mean = np.mean(dataset.data, axis=0)
        self._std = np.std(dataset.data, axis=0)
        normalized = (dataset.data - self._mean) / self._std
        self._q = np.average(np.average(normalized, axis=1) / dataset.target)

    def predict(self, dataset):
        normalized = (dataset.data - self._mean) / self._std
        predicted = np.average(normalized, axis=1) / self._q
        return predicted


class LinearRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the linear regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Linear regressor.
        """
        regressor = LinearRegression(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()

        return param_space


class RidgeRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the ridge regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Ridge regressor.
        """
        regressor = Ridge(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        # param_space['alpha'] = ?
        param_space['fit_intercept'] = BooleanParameter()
        param_space['max_iter'] = ChoiceParameter([None])  # ?
        param_space['normalize'] = BooleanParameter()
        param_space['solver'] = ChoiceParameter(['auto', 'svd', 'dense_cholesky', 'lsqr',
                                                 'sparse_cg'])
        param_space['tol'] = ChoiceParameter([0.001])

        return param_space


class RidgeCVRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the ridge regressor class (built-in cross-validation), with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Ridge regressor (CV).
        """
        regressor = RidgeCV(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        # param_space['alphas'] = ?
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        param_space['scoring'] = ChoiceParameter(['r2', 'mean_squared_error', 'accuracy', 'f1',
                                                  'roc_auc', 'average_precision', 'precision',
                                                  'recall', 'log_loss', 'adjusted_rand_score'])
        # param_space['cv'] = ?
        param_space['gcv_mode'] = ChoiceParameter(['auto', 'svd', 'eigen'])

        return param_space


class LassoRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Lasso regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Lasso regressor.
        """
        regressor = Lasso(**parameters)
        return regressor

    def train(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(LassoRegressor, self).train(dataset)
            return trained

    def predict(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predict = super(LassoRegressor, self).predict(dataset)
            return predict

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['alpha'] = ChoiceParameter([1.0])
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])
        param_space['max_iter'] = ChoiceParameter([10000])
        param_space['tol'] = ChoiceParameter([0.001])
        param_space['warm_start'] = BooleanParameter()
        param_space['positive'] = BooleanParameter()

        return param_space


class LassoCVRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the lasso regressor class (built-in cross-validation), with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Lasso regressor (CV).
        """
        regressor = LassoCV(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['eps'] = ChoiceParameter([0.001])
        param_space['n_alphas'] = ChoiceParameter([100])
        # param_space['alphas'] = ?
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])
        param_space['max_iter'] = ChoiceParameter([1000])
        param_space['tol'] = ChoiceParameter([0.0001])
        # param_space['cv'] = ?

        return param_space


class LassoLarsRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Lasso-LARS regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Lasso-LARS regressor.
        """
        regressor = LassoLars(**parameters)
        return regressor

    def train(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(LassoLarsRegressor, self).train(dataset)
            return trained

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['alpha'] = ChoiceParameter([1.0])
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])
        param_space['max_iter'] = ChoiceParameter([500])
        param_space['fit_path'] = BooleanParameter()

        return param_space


class LassoLarsCVRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the lasso-LARS regressor class (built-in cross-validation), with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Lasso-LARS regressor (CV).
        """
        regressor = LassoLarsCV(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])
        param_space['max_iter'] = ChoiceParameter([1000])
        # param_space['cv'] = ?
        param_space['max_n_alphas'] = ChoiceParameter([1000])

        return param_space


class LarsRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the LARS regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: LARS regressor.
        """
        regressor = Lars(**parameters)
        return regressor

    def train(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(LarsRegressor, self).train(dataset)
            return trained

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_nonzero_coefs'] = ChoiceParameter([500])
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])
        param_space['fit_path'] = BooleanParameter()

        return param_space


class LarsCVRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the LARS regressor class (built-in cross-validation), with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: LARS regressor (CV).
        """
        regressor = LarsCV(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])
        param_space['max_iter'] = ChoiceParameter([500])
        # param_space['cv'] = ?
        param_space['max_n_alphas'] = ChoiceParameter([1000])

        return param_space


class ElasticNetRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Elastic net regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Elastic net regressor.
        """
        regressor = ElasticNet(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['alpha'] = ChoiceParameter([1.0])
        param_space['l1_ratio'] = ChoiceParameter([0.5])
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])
        param_space['max_iter'] = ChoiceParameter([1000])
        param_space['tol'] = ChoiceParameter([0.0001])
        param_space['warm_start'] = BooleanParameter()  # ?
        param_space['positive'] = BooleanParameter()

        return param_space

    def train(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(ElasticNetRegressor, self).train(dataset)
            return trained


class ElasticNetCVRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Elastic net regressor class (built-in cross-validation), with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Elastic net regressor (CV).
        """
        regressor = ElasticNetCV(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['l1_ratio'] = ChoiceParameter([0.5])
        param_space['eps'] = ChoiceParameter([0.001])
        param_space['n_alphas'] = ChoiceParameter([100])
        # param_space['alphas'] = ?
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])
        param_space['max_iter'] = ChoiceParameter([1000])
        param_space['tol'] = ChoiceParameter([0.0001])
        # param_space['cv'] = ?

        return param_space


class OrthogonalMatchingPursuitRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the orthogonal matching pursuit regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Orthogonal matching pursuit regressor.
        """
        regressor = OrthogonalMatchingPursuit(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['tol'] = ChoiceParameter([None])
        param_space['n_nonzero_coefs'] = ChoiceParameter([None])
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()
        param_space['precompute'] = ChoiceParameter([True, False, 'auto'])

        return param_space


class BayesianRidgeRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Bayesian ridge regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Bayesian ridge regressor.
        """
        regressor = BayesianRidge(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_iter'] = ChoiceParameter([300])
        param_space['tol'] = ChoiceParameter([0.001])
        param_space['alpha_1'] = ChoiceParameter([0.000001])
        param_space['alpha_2'] = ChoiceParameter([0.000001])
        param_space['lambda_1'] = ChoiceParameter([0.000001])
        param_space['lambda_2'] = ChoiceParameter([0.000001])
        param_space['compute_score'] = BooleanParameter()
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()

        return param_space


class ARDRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Bayesian ARD regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Bayesian ARD regressor.
        """
        regressor = ARDRegression(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_iter'] = ChoiceParameter([300])
        param_space['tol'] = ChoiceParameter([0.001])
        param_space['alpha_1'] = ChoiceParameter([0.000001])
        param_space['alpha_2'] = ChoiceParameter([0.000001])
        param_space['lambda_1'] = ChoiceParameter([0.000001])
        param_space['lambda_2'] = ChoiceParameter([0.000001])
        param_space['compute_score'] = BooleanParameter()
        param_space['threshold_lambda'] = ChoiceParameter([10000.0])
        param_space['fit_intercept'] = BooleanParameter()
        param_space['normalize'] = BooleanParameter()

        return param_space


class SGDRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Stochastic gradient descent regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Stochastic gradient descent regressor.
        """
        regressor = SKSGDRegressor(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['loss'] = ChoiceParameter(['squared_loss'])
        param_space['penalty'] = ChoiceParameter(['l2'])
        param_space['alpha'] = ChoiceParameter([0.0001])
        param_space['l1_ratio'] = ChoiceParameter([0.15])
        param_space['fit_intercept'] = ChoiceParameter([True])
        param_space['n_iter'] = ChoiceParameter([5])
        param_space['shuffle'] = ChoiceParameter([False])
        # param_space['random_state'] = ?
        param_space['epsilon'] = ChoiceParameter([0.1])
        param_space['learning_rate'] = ChoiceParameter(['constant'])
        param_space['power_t'] = ChoiceParameter([0.25])
        param_space['eta0'] = ChoiceParameter([0.01])
        param_space['warm_start'] = ChoiceParameter([False])

        #param_space['loss'] = ChoiceParameter(['squared_loss', 'huber', 'epsilon_insensitive',
        #                                       'squared_epsilon_insensitive'])
        #param_space['penalty'] = ChoiceParameter(['l2', 'l1', 'elasticnet'])
        #param_space['alpha'] = ChoiceParameter([0.0001])
        #param_space['l1_ratio'] = ChoiceParameter([0.15])
        #param_space['fit_intercept'] = BooleanParameter()
        #param_space['n_iter'] = ChoiceParameter([5])
        #param_space['shuffle'] = BooleanParameter()
        ## param_space['random_state'] = ?
        #param_space['epsilon'] = ChoiceParameter([0.1])
        #param_space['learning_rate'] = ChoiceParameter(['constant', 'optimal', 'invscaling'])
        #param_space['power_t'] = ChoiceParameter([0.25])
        #param_space['eta0'] = ChoiceParameter([0.01])
        #param_space['warm_start'] = BooleanParameter()

        return param_space


class PassiveAggressiveRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the passive aggressive regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Passive aggressive regressor.
        """
        regressor = SKPassiveAggressiveRegressor(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['C'] = ChoiceParameter([1.0])
        param_space['epsilon'] = ChoiceParameter([0.1])
        param_space['fit_intercept'] = BooleanParameter()
        param_space['n_iter'] = ChoiceParameter([5])
        param_space['shuffle'] = BooleanParameter()
        # param_space['random_state'] = ?
        param_space['loss'] = ChoiceParameter(['epsilon_insensitive', 'squared_epsilon_insensitive'])
        param_space['warm_start'] = BooleanParameter()

        return param_space


class SVMRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the support vector machine regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Support vector machine regressor.
        """
        regressor = SVR(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['C'] = ChoiceParameter([1.0])
        param_space['epsilon'] = ChoiceParameter([0.1])
        param_space['kernel'] = ChoiceParameter(['rbf']) #, 'linear', 'poly', 'sigmoid',
                                                 #'precomputed'])
        param_space['degree'] = ChoiceParameter([3])  # ?
        param_space['coef0'] = ChoiceParameter([0.0])
        param_space['gamma'] = ChoiceParameter([0.0])
        param_space['shrinking'] = BooleanParameter()
        param_space['tol'] = ChoiceParameter([1e-3])
        # param_space['random_state'] = ?

        return param_space


class NuSVMRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Nu Support vector machine regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Nu Support vector machine regressor.
        """
        regressor = NuSVR(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['C'] = ChoiceParameter([1.0])
        param_space['nu'] = ChoiceParameter([0.5])
        param_space['kernel'] = ChoiceParameter(['linear', 'poly', 'rbf', 'sigmoid',
                                                 'precomputed'])
        param_space['degree'] = ChoiceParameter([3])  # ?
        param_space['coef0'] = ChoiceParameter([0.0])
        param_space['gamma'] = ChoiceParameter([0.0])
        param_space['shrinking'] = BooleanParameter()
        param_space['tol'] = ChoiceParameter([1e-3])
        # param_space['random_state'] = ?

        return param_space


class KNNRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the k-Nearest-Neighbors regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: K Nearest-Neighbors regressor.
        """
        try:
            regressor = KNeighborsRegressor(**parameters)
            return regressor
        except Exception as e:
            pass
            #print("{0} made it fail".format(parameters))

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_neighbors'] = ChoiceParameter([5])
        param_space['weights'] = ChoiceParameter(['uniform', 'distance'])
        param_space['algorithm'] = ChoiceParameter(['brute', 'ball_tree', 'kd_tree'])
        param_space['leaf_size'] = ChoiceParameter([30])  # ???
        param_space['p'] = ChoiceParameter([2])
        # param_space['metric'] = ChoiceParameter(['euclidean', 'manhattan', 'chebyshev',
        #                                         'minkowski', 'wminkowski', 'seuclidean',
        #                                         'mahalanobis'])
        param_space['metric'] = ChoiceParameter(['euclidean', 'manhattan', 'chebyshev',
                                                 'minkowski', 'wminkowski',
                                                 'seuclidean',
                                                 'mahalanobis'
                                                ])


        return param_space


class RadiusNeighborsRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Radius Neighbors regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Radius Neighbors regressor.
        """
        regressor = SKRadiusNeighborsRegressor(**parameters)
        return regressor

    def predict(self, dataset):
        with warnings.catch_warnings():
            # Warnings expected when no samples were found within the
            # neighborhood radius. TODO find a way to inform SALT to increase
            # the radius.
            warnings.simplefilter("ignore")
            prediction = super(RadiusNeighborsRegressor, self).predict(dataset)
            if not np.all(np.isfinite(prediction)):
                raise Exception
            return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['radius'] = ChoiceParameter([100000.0])
        param_space['weights'] = ChoiceParameter(['uniform', 'distance'])
        param_space['algorithm'] = ChoiceParameter(['brute', 'ball_tree', 'kd_tree'])
        param_space['leaf_size'] = ChoiceParameter([30])  # ???
        param_space['metric'] = ChoiceParameter(['euclidean', 'manhattan', 'chebyshev',
                                                 'minkowski', 'wminkowski', 'seuclidean',
                                                 'mahalanobis'])

        param_space['p'] = ChoiceParameter([2])
        '''
        param_space['radius'] = ChoiceParameter([100.0])
        param_space['weights'] = ChoiceParameter(['uniform'])
        param_space['algorithm'] = ChoiceParameter(['brute'])
        param_space['leaf_size'] = ChoiceParameter([30])  # ???
        param_space['metric'] = ChoiceParameter(['euclidean'])

        param_space['p'] = ChoiceParameter([2])
        '''
        return param_space


class GaussianProcessRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the Gaussian process regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Gaussian process regressor.
        """
        regressor = GaussianProcess(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['regr'] = ChoiceParameter(['constant', 'linear', 'quadratic'])  # breaks
        param_space['corr'] = ChoiceParameter(['absolute_exponential', 'squared_exponential',  # breaks
                                               'generalized_exponential', 'cubic', 'linear'])
        # param_space['beta0'] = ?
        param_space['storage_mode'] = ChoiceParameter(['full', 'light'])
        #param_space['theta0'] = ChoiceParameter([None])
        #param_space['thetaL'] = ChoiceParameter([None])
        #param_space['thetaU'] = ChoiceParameter([None])
        param_space['normalize'] = BooleanParameter()
        #param_space['nugget'] = BooleanParameter() ???
        param_space['optimizer'] = ChoiceParameter(['fmin_cobyla', 'Welch'])
        #param_space['random_start'] = ChoiceParameter([1])

        return param_space


class DecisionTreeRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the decision tree regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Decision tree regressor.
        """
        regressor = SKDesicionTreeRegressor(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['criterion'] = ChoiceParameter(['mse'])
        #param_space['max_features'] = ?
        param_space['max_depth'] = ChoiceParameter([None])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        #param_space['random_state'] = ?
        return param_space


class RandomForestRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the random forest regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Random forest regressor.
        """
        regressor = SKRandomForestRegressor(**parameters)
        return regressor

    def train(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(RandomForestRegressor, self).train(dataset)
            return trained

    def predict(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = super(RandomForestRegressor, self).predict(dataset)
            return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_estimators'] = ChoiceParameter([10])
        param_space['criterion'] = ChoiceParameter(['mse'])
        #param_space['max_features'] = ?
        param_space['max_depth'] = ChoiceParameter([None])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        param_space['bootstrap'] = ChoiceParameter([True])
        param_space['oob_score'] = BooleanParameter()
        #param_space['random_state'] = ?

        return param_space


class ExtraTreeEnsembleRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the extra-trees regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Extra-trees regressor.
        """
        regressor = ExtraTreesRegressor(**parameters)
        return regressor

    def train(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = super(ExtraTreeEnsembleRegressor, self).train(dataset)
            return trained

    def predict(self, dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = super(ExtraTreeEnsembleRegressor, self).predict(dataset)
            return prediction

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['n_estimators'] = ChoiceParameter([10])
        param_space['criterion'] = ChoiceParameter(['mse'])
        #param_space['max_features'] = ?
        param_space['max_depth'] = ChoiceParameter([None])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        param_space['bootstrap'] = ChoiceParameter([True])
        param_space['oob_score'] = BooleanParameter()
        #param_space['random_state'] = ?

        return param_space


class GradientBoostingRegressor(BaseRegressor):
    def create_regressor(self, **parameters):
        """Creates the gradient boosting regressor class, with the parameters given.

        :param parameters: parameters to pass to the constructor.
        :returns: Gradient boosting regressor.
        """
        regressor = SKGradientBoostingRegressor(**parameters)
        return regressor

    @classmethod
    def create_default_params(self):
        param_space = ParameterSpace()

        param_space['loss'] = ChoiceParameter(['ls', 'lad', 'huber', 'quantile'])
        param_space['learning_rate'] = ChoiceParameter([0.1])
        param_space['n_estimators'] = ChoiceParameter([100])
        param_space['max_depth'] = ChoiceParameter([3])  # ?
        param_space['min_samples_split'] = ChoiceParameter([2])  # ?
        param_space['min_samples_leaf'] = ChoiceParameter([1])  # ?
        param_space['subsample'] = ChoiceParameter([1.0])  # ?
        # param_space['max_features'] = ?
        # param_space['init'] = ?

        return param_space
