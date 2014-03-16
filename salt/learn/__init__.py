"""
The :mod:`salt.learn` subpackage implements the classes used to perform training and prediction
on classification, regression, and clustering problems.

.. todo::

    Register learners dynamically.

"""

from .base import create_parameter_space
from .classifiers import (LogisticRegressionClassifier, SGDClassifier,
                          PassiveAggressiveClassifier, RidgeClassifier, RidgeCVClassifier,
                          GaussianNBClassifier, MultinomialNBClassifier, BernoulliNBClassifier,
                          KNNClassifier, RadiusNeighborsClassifier, NearestCentroidClassifier,
                          DecisionTreeClassifier, ExtraTreeClassifier, RandomForestClassifier,
                          ExtraTreeEnsembleClassifier, GradientBoostingClassifier,
                          GaussianProcessClassifier, SVMClassifier, LinearSVMClassifier,
                          NuSVMClassifier, LinearDiscriminantClassifier,
                          QuadraticDiscriminantClassifier, )
from .regressors import (LinearRegressor, RidgeRegressor, RidgeCVRegressor, LassoRegressor,
                         LassoCVRegressor, LassoLarsRegressor, LassoLarsCVRegressor,
                         LarsRegressor, LarsCVRegressor, ElasticNetRegressor,
                         ElasticNetCVRegressor, OrthogonalMatchingPursuitRegressor,
                         BayesianRidgeRegressor, ARDRegressor, SGDRegressor,
                         PassiveAggressiveRegressor, SVMRegressor, NuSVMRegressor, KNNRegressor,
                         RadiusNeighborsRegressor, GaussianProcessRegressor,
                         DecisionTreeRegressor, RandomForestRegressor,
                         ExtraTreeEnsembleRegressor, GradientBoostingRegressor, )

__all__ = ['AVAILABLE_CLASSIFIERS', 'AVAILABLE_REGRESSORS', 'DEFAULT_CLASSIFIERS',
           'DEFAULT_REGRESSORS', 'create_parameter_space']

AVAILABLE_CLASSIFIERS = {'LogisticRegressionClassifier': LogisticRegressionClassifier,
                         'SGDClassifier': SGDClassifier,
                         'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
                         'RidgeClassifier': RidgeClassifier,
                         'ridgecv': RidgeCVClassifier,
                         'GaussianNBClassifier': GaussianNBClassifier,
                         'multinomial': MultinomialNBClassifier,
                         'bernoulinb': BernoulliNBClassifier,
                         'KNNClassifier': KNNClassifier,
                         'RadiusNeighborsClassifier': RadiusNeighborsClassifier,
                         'NearestCentroidClassifier': NearestCentroidClassifier,
                         'DecisionTreeClassifier': DecisionTreeClassifier,
                         'extratree': ExtraTreeClassifier,
                         'RandomForestClassifier': RandomForestClassifier,
                         'ExtraTreeEnsembleClassifier': ExtraTreeEnsembleClassifier,
                         'GradientBoostingClassifier': GradientBoostingClassifier,
                         'GaussianProcessClassifier': GaussianProcessClassifier,  # Remove?
                         'SVMClassifier': SVMClassifier,
                         'LinearSVMClassifier': LinearSVMClassifier,
                         'NuSVMClassifier': NuSVMClassifier,
                         'LinearDiscriminantClassifier': LinearDiscriminantClassifier,
                         'QuadraticDiscriminantClassifier': QuadraticDiscriminantClassifier,
                         }
AVAILABLE_REGRESSORS = {'linear': LinearRegressor,
                        'ridge': RidgeRegressor,
                        'ridgecv': RidgeCVRegressor,
                        'lasso': LassoRegressor,
                        'lassocv': LassoCVRegressor,
                        'lassolars': LassoLarsRegressor,
                        'lassolarscv': LassoLarsCVRegressor,
                        'lars': LarsRegressor,
                        'larscv': LarsCVRegressor,
                        'elasticnet': ElasticNetRegressor,
                        'elasticnetcv': ElasticNetCVRegressor,
                        'orthmp': OrthogonalMatchingPursuitRegressor,
                        'bayesianridge': BayesianRidgeRegressor,
                        'ard': ARDRegressor,
                        'sgd': SGDRegressor,
                        'passiveaggressive': PassiveAggressiveRegressor,
                        'svm': SVMRegressor,
                        'nusvm': NuSVMRegressor,
                        'knn': KNNRegressor,
                        'radius': RadiusNeighborsRegressor,
                        'gaussianprocess': GaussianProcessRegressor,
                        'decisiontree': DecisionTreeRegressor,
                        'randomforest': RandomForestRegressor,
                        'extratrees': ExtraTreeEnsembleRegressor,
                        'gradientboosting': GradientBoostingRegressor,
                        }

DEFAULT_CLASSIFIERS = [#LogisticRegressionClassifier,
                       #SGDClassifier,
                       #PassiveAggressiveClassifier,
                       #RidgeClassifier,
                       ##RidgeCVClassifier,
                       #GaussianNBClassifier,
                       ##MultinomialNBClassifier,
                       ##BernoulliNBClassifier,
                       #KNNClassifier,
                       #RadiusNeighborsClassifier,
                       #NearestCentroidClassifier,
                       #DecisionTreeClassifier,
                       ##ExtraTreeClassifier,
                       #RandomForestClassifier,
                       #ExtraTreeEnsembleClassifier,
                       #GradientBoostingClassifier,
                       #GaussianProcessClassifier,  # Remove?
                       #SVMClassifier,
                       #LinearSVMClassifier,
                       #NuSVMClassifier,
                       LinearDiscriminantClassifier,
                       QuadraticDiscriminantClassifier,
                      ]

DEFAULT_REGRESSORS = [LinearRegressor,
                      RidgeRegressor,
                      LassoRegressor,
                      LassoLarsRegressor,
                      LarsRegressor,
                      ElasticNetRegressor,
                      OrthogonalMatchingPursuitRegressor,
                      BayesianRidgeRegressor,
                      ARDRegressor,
                      SGDRegressor,  # bug?
                      PassiveAggressiveRegressor,
                      #SVMRegressor,  # slow
                      #NuSVMRegressor,  # bug?
                      KNNRegressor,
                      RadiusNeighborsRegressor,
                      GaussianProcessRegressor,
                      DecisionTreeRegressor,
                      RandomForestRegressor,
                      ExtraTreeEnsembleRegressor,
                      GradientBoostingRegressor
                      ]
