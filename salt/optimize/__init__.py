"""The :mod:`salt.optimize` subpackage implements optimization techniques."""

from .optimize import KDEOptimizer, ShrinkingHypercubeOptimizer, DefaultConfigOptimizer

__all__ = ['KDEOptimizer', 'ShrinkingHypercubeOptimizer', 'DefaultConfigOptimizer', 'AVAILABLE_OPTIMIZERS']

AVAILABLE_OPTIMIZERS = {
    'randomsearch': KDEOptimizer,
    'kde': KDEOptimizer,
    'shrinking': ShrinkingHypercubeOptimizer,
    'none': DefaultConfigOptimizer
}
