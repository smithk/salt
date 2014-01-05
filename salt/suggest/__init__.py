"""
The :mod:`salt.suggest` subpackage implements classes to analyze, evaluate, and rank
learning techniques, and suggest best configurations.
"""

from .suggest import SuggestionTaskManager
from .rank import rank

__all__ = ['SuggestionTaskManager','rank']
