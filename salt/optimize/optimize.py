"""The :mod:`salt.optimize.base` module provides classes for optimization."""

from bisect import insort
from numpy.random import shuffle
from six.moves import range


class BaseOptimizer(object):
    def __init__(self, param_space):
        self.param_space = param_space
        self.evaluation_results = []
        self.evaluation_parameters = []

    def add_results(self, evaluation_results):
        insort(self.evaluation_results, evaluation_results)

    def get_next_configuration(self):
        raise NotImplementedError

    @staticmethod
    def get_best_configs(optimizer_list, top_n=5):
        configs = {optimizer.evaluation_results[0].learner: optimizer.evaluation_results[:top_n]
                   for optimizer in optimizer_list if len(optimizer.evaluation_results) > 0}
        return configs
        '''
        return evaluation_results[:top_n]
        all_results = sorted(itertools.chain.from_iterable(ranking.values()), reverse=True)
        Log.write_color("\n=========================== RESULTS ===========================", 'OKGREEN')
        print('')
        print("Global ranking: (top {0})".format(top_n))
        for result in all_results[:top_n]:
            print(string_template.format(score=result.metrics.score,
                                        learner=result.learner,
                                        parameters=result.parameters))

        print("\nLearner-wise ranking: (top {0} per learner)".format(top_n))
        for learner in ranking.keys():
            print("- {learner}:".format(learner=learner))
            learner_results = sorted(ranking[learner], reverse=True)
            for result in learner_results[:TOP]:
                print(string_template.format(score=result.metrics.score,
                                            learner=learner,
                                            parameters=result.parameters))
        '''


class SequentialOptimizer(BaseOptimizer):
    def __init__(self, param_space):
        super(SequentialOptimizer, self).__init__(param_space)
        self.parameter_list = list(param_space.get_grid())
        param_space_size = len(self.parameter_list)
        self.indices = list(range(param_space_size - 1, -1, -1))

    def get_next_configuration(self):
        if not self.indices:
            return

        next_configuration = self.parameter_list[self.indices.pop()]
        return next_configuration


class RandomOptimizer(SequentialOptimizer):
    def __init__(self, param_space):
        super(RandomOptimizer, self).__init__(param_space)
        shuffle(self.indices)


class KDEOptimizer(BaseOptimizer):
    def __init__(self, param_space):
        super(KDEOptimizer, self).__init__(param_space)

    def get_next_configuration(self):
        next_configuration = self.param_space.sample_configuration()
        return next_configuration
