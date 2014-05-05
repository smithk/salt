"""The :mod:`salt.optimize` subpackage implements optimization techniques."""

from .optimize import KDEOptimizer, ShrinkingHypercubeOptimizer, DefaultConfigOptimizer, StaticConfigListOptimizer

__all__ = ['KDEOptimizer', 'ShrinkingHypercubeOptimizer', 'DefaultConfigOptimizer',
           'StaticConfigListOptimizer', 'SPECIAL_OPTIMIZERS', 'AVAILABLE_OPTIMIZERS']

AVAILABLE_OPTIMIZERS = {
    'randomsearch': KDEOptimizer,
    'kde': KDEOptimizer,
    'shrinking': ShrinkingHypercubeOptimizer,
    'none': DefaultConfigOptimizer
}

SPECIAL_OPTIMIZERS = {'list': StaticConfigListOptimizer}
