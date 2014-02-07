"""The :mod:`salt.optimize.base` module provides classes for optimization."""

from bisect import insort
from numpy.random import shuffle
import numpy as np
from six import iteritems
from six.moves import range
from ..parameters.param import Uniform, LogUniform, Normal, LogNormal, ParameterSpace


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


class ShrinkingHypercubeOptimizer(BaseOptimizer):
    def __init__(self, param_space):
        super(ShrinkingHypercubeOptimizer, self).__init__(param_space)
        self.best_results = {}
        self.hypercubes = {}
        self.expand_rate = 2.0
        self.shrink_rate = 0.98
        self.hypercube_threshold = 1e-4

    @classmethod
    def create_hypercube(self, numerical_params):
        starting_width = 0.05  # percentage of the param space range
        hypercube = {}
        for name, param in iteritems(numerical_params):
            dist = param.distribution
            if type(dist) is Uniform:
                hypercube[name] = starting_width * (dist.upper - dist.lower)
            elif type(dist) is LogUniform:
                hypercube[name] = starting_width * (dist.upper - dist.lower)
            elif type(dist) is Normal:
                hypercube[name] = starting_width * dist.stdev
            elif type(dist) is LogNormal:
                hypercube[name] = starting_width * dist.stdev
            else:
                print(name, ' is unknown distribution type')
        return hypercube

    def get_hypercube(self, configuration):
        signature = ParameterSpace.get_cat_signature(self.param_space, configuration)
        hypercube = self.hypercubes.get(str(signature))
        if hypercube is None:
            numerical_params = ParameterSpace.get_numerical_space(self.param_space, configuration)
            hypercube = ShrinkingHypercubeOptimizer.create_hypercube(numerical_params)
            self.hypercubes[str(signature)] = hypercube
        return hypercube

    def add_results(self, evaluation_results):
        super(ShrinkingHypercubeOptimizer, self).add_results(evaluation_results)
        signature = ParameterSpace.get_cat_signature(self.param_space, evaluation_results.parameters)
        best_result = self.best_results.get(str(signature))
        if best_result is None:  # and len(self.evaluation_results) > 0:  # TODO Parametrize this
            best_result = evaluation_results

        #if self.best_result is not None:
        if evaluation_results.metrics.score >= best_result.metrics.score:
            best_result = evaluation_results
            self.expand(best_result.parameters)
            self.best_results[str(signature)] = best_result
        else:
            self.shrink(best_result.parameters)  # Shrink around same config
        #else:
        #    self.best_result = evaluation_results

    def expand(self, configuration, rate=None):
        expand_rate = rate or self.expand_rate
        hypercube = self.get_hypercube(configuration)
        numerical_params = ParameterSpace.get_numerical_space(self.param_space, configuration)
        for name, param in iteritems(numerical_params):
            dist = param.distribution
            basedist = param.prior
            if hypercube[name] > self.hypercube_threshold:
                hypercube[name] *= expand_rate
            if type(dist) is Uniform:
                if type(basedist) is Uniform:  # Strong boundaries
                    dist.lower = max(basedist.lower, configuration[name] - hypercube[name] / 2.0)
                    dist.upper = min(basedist.upper, configuration[name] + hypercube[name] / 2.0)
                else:
                    dist.lower = configuration[name] - hypercube[name] / 2.0
                    dist.upper = configuration[name] + hypercube[name] / 2.0
            elif type(dist) is LogUniform:
                if type(basedist) is LogUniform:  # Strong boundaries
                    dist.lower = max(basedist.lower, np.log(configuration[name]) - hypercube[name] / 2.0)
                    dist.upper = min(basedist.upper, np.log(configuration[name]) + hypercube[name] / 2.0)
                else:
                    dist.lower = np.log(configuration[name]) - hypercube[name] / 2.0
                    dist.upper = np.log(configuration[name]) + hypercube[name] / 2.0
            elif type(dist) is Normal:
                param.distribution = Uniform(lower=configuration[name] - hypercube[name] / 2.0,
                                             upper=configuration[name] + hypercube[name] / 2.0)
            elif type(dist) is LogNormal:
                param.distribution = LogUniform(lower=np.log(configuration[name]) - hypercube[name] / 2.0,
                                                upper=np.log(configuration[name]) + hypercube[name] / 2.0)

    def shrink(self, configuration):
        self.expand(configuration, self.shrink_rate)

    def get_next_configuration(self):
        if len(self.hypercubes) > 0 and all(value <= self.hypercube_threshold for hypercube in self.hypercubes.values() for value in hypercube.values()):
            next_configuration = None
            a = [(result.learner, result.parameters.values(), result.metrics.score) for result in self.evaluation_results]
            with open('/tmp/results.txt', 'w') as f:
                for i in a:
                    f.write("{0}, {1}, {2}\n".format(*i))
        else:
            #print([value for hypercube in self.hypercubes.values() for value in hypercube.values()])
            next_configuration = self.param_space.sample_configuration()
        return next_configuration


class DefaultConfigOptimizer(BaseOptimizer):
    def get_next_configuration(self):
        next_configuration = {}
        return next_configuration
