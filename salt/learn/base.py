"""
The :mod:`salt.learn.base` module includes base classes to describe a
learning implementation.
"""


class BaseLearner(object):
    def _get_name(self):
        return type(self).__name__

    learner_name = property(_get_name)


def create_parameter_space(learners, settings):
    '''Create the parameter space structure for each learner from the list,
    according to the given options.'''
    parameter_dict = {learner.__name__: learner.create_param_space(settings) for learner in learners}
    return parameter_dict
