"""
The :mod:`salt.learn.base` module includes base classes to describe a
learning implementation.
"""


class BaseLearner(object):
    def _get_name(self):
        return type(self).__name__

    learner_name = property(_get_name)
