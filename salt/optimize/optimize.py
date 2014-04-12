"""The :mod:`salt.optimize.base` module provides classes for optimization."""

#from bisect import insort_left as insort
from numpy.random import shuffle
import numpy as np
from six import iteritems
from six.moves import range
from ..parameters.param import Uniform, LogUniform, Normal, LogNormal, ParameterSpace
from ..evaluate.evaluate import ResultSet
from copy import deepcopy
import cPickle


class BaseOptimizer(object):
    def __init__(self, param_space):
        self.best = None
        self.param_space = param_space
        self.evaluation_results = []
        self.evaluation_parameters = []
        self.initial_configurations = [{}]  # Try these first

    def add_results(self, evaluation_results):
        #insort(self.evaluation_results, evaluation_results)
        self.evaluation_results.append(evaluation_results)
        with open("data/{0}_all".format(evaluation_results.learner), 'a') as output:
            cPickle.dump((evaluation_results._mean, evaluation_results.runtime, evaluation_results.configuration), output)
        if evaluation_results > self.best:
            self.best = evaluation_results

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
    '''Explore a finite parameter space sequentially.'''
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
    '''Explore a finite parameter space randomly.'''
    def __init__(self, param_space):
        super(RandomOptimizer, self).__init__(param_space)
        shuffle(self.indices)


class DefaultConfigOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super(DefaultConfigOptimizer, self).__init__(*args, **kwargs)
        self.has_run = False

    def get_next_configuration(self):
        if self.has_run:
            next_configuration = None
        else:
            next_configuration = {}
            self.has_run = True
        return next_configuration


class KDEOptimizer(BaseOptimizer):
    def __init__(self, param_space):
        super(KDEOptimizer, self).__init__(param_space)
        self.configurations = [{}]
        numerical_params = param_space.numerical_params
        self.numerical_params_exist = len(numerical_params) > 0
        # TODO Do this correctly

    def configurations_exhausted(self):
        print("TESTING IF THE PARAMETER SPACE HAS BEEN EXHAUSTED")
        return not self.numerical_params_exist

    def get_next_configuration(self):
        if len(self.initial_configurations) > 0:
            next_configuration = self.initial_configurations.pop()
        else:
            next_configuration = self.param_space.sample_configuration()
            while next_configuration in self.configurations:
                #if self.configurations_exhausted():
                #    next_configuration = None
                #    break
                next_configuration = self.param_space.sample_configuration()
            self.configurations.append(next_configuration)
        return next_configuration


class ModelEstimationOptimizer(BaseOptimizer):
    def __init__(self, param_space):
        super(ModelEstimationOptimizer, self).__init__(param_space)
        self.p_prior = 1.
        self.p_learned = 0.

    def add_results(self, evaluation_results):
        super(ModelEstimationOptimizer, self).add_results(evaluation_results)


class ShrinkingHypercubeOptimizer(BaseOptimizer):
    def __init__(self, param_space):
        super(ShrinkingHypercubeOptimizer, self).__init__(param_space)
        #self.best_results = {}
        self.hypercubes = {}  # One hypercube for each signature
        self.expand_rate = 1.2
        self.shrink_rate = 0.97
        self.hypercube_threshold = 1e-4
        self.configurations = []
        default = param_space.get_default()
        print("default settings are: ", default)
        self.initial_configs = []  # [default]  # Try first with default configuration
        numerical_params = param_space.numerical_params
        self.numerical_params_exist = len(numerical_params) > 0
        # TODO Do this correctly
        #self.best_result_groups = []
        #self.hypercube_results = {}

        self.hypercube_bests = {}
        self.num_configs_tried = 0

    def configurations_exhausted(self):
        return not self.numerical_params_exist

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
        #print("created hypercube: {0}".format(hypercube))
        return hypercube

    def get_hypercube(self, configuration):
        signature = ParameterSpace.get_cat_signature(self.param_space, configuration)
        hypercube = self.hypercubes.get(str(signature))
        if hypercube is None:
            numerical_params = ParameterSpace.get_numerical_space(self.param_space, configuration)
            hypercube = ShrinkingHypercubeOptimizer.create_hypercube(numerical_params)
            self.hypercubes[str(signature)] = hypercube
        return hypercube

    def get_score_list(self, configuration, hypercube_score_list):
        # Obtain the score list for a specific configuration in their
        # respective hypercube (the one matching its signature)
        score_list = hypercube_score_list.get(str(configuration))
        if score_list is None:
            score_list = ResultSet(configuration)
            hypercube_score_list[str(configuration)] = score_list
        return score_list

    def add_results(self, evaluation_results):
        super(ShrinkingHypercubeOptimizer, self).add_results(evaluation_results)

        hypercube_signature = ParameterSpace.get_cat_signature(self.param_space, evaluation_results.configuration)
        hypercube_signature = str(hypercube_signature)
        hypercube_best = self.hypercube_bests.get(hypercube_signature)
        if hypercube_best is None:
            self.hypercube_bests[hypercube_signature] = evaluation_results
        if evaluation_results > hypercube_best:
            self.expand(evaluation_results.configuration)
            self.hypercube_bests[hypercube_signature] = evaluation_results
        else:
            self.shrink(hypercube_best.configuration)

    def expand(self, configuration, rate=None):
        expand_rate = rate or self.expand_rate
        hypercube = self.get_hypercube(configuration)
        numerical_params = ParameterSpace.get_numerical_space(self.param_space, configuration)
        can_shrink = len(numerical_params) == 0
        for name, param in iteritems(numerical_params):
            dist = param.distribution
            basedist = param.prior
            if hypercube[name] > self.hypercube_threshold:
                hypercube[name] *= expand_rate
                can_shrink = True
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
        #print("hypercube size: {0}".format(hypercube))
        if not can_shrink:
            self.reset_hypercube(configuration)

    def reset_hypercube(self, configuration):
        signature = ParameterSpace.get_cat_signature(self.param_space, configuration)
        numerical_params = ParameterSpace.get_numerical_space(self.param_space, configuration)
        print("CAN'T SHRINK HYPERCUBE {0} for {1}, RESETTING".format(signature, self.param_space.name))
        for name, param in iteritems(numerical_params):
            print("restoring distribution {0}".format(param.prior))
            param.distribution = deepcopy(param.prior)
        if str(signature) in self.hypercubes:
            del self.hypercubes[str(signature)]

    def shrink(self, configuration):
        self.expand(configuration, self.shrink_rate)

    def get_next_configuration(self):
        if len(self.initial_configs) > 0:
            next_configuration = self.initial_configs.pop()
        else:
            if len(self.hypercubes) > 0 and all(value <= self.hypercube_threshold for hypercube in self.hypercubes.values() for value in hypercube.values()):
                self.hypercubes = {}
                next_configuration = self.param_space.sample_configuration()
                #next_configuration = None  # TODO Restart hypercube on different initial conditions
            else:
                next_configuration = self.param_space.sample_configuration()
                while next_configuration in self.configurations:
                    print("configuration {0} has already been sampled".format(next_configuration))
                    #if self.configurations_exhausted():  # TODO Check exhaustion properly
                    #    next_configuration = None
                    #    break
                    self.reset_hypercube(next_configuration)
                    next_configuration = self.param_space.sample_configuration()
            self.configurations.append(next_configuration)
        self.num_configs_tried += 1
        #print("now trying {0} ({1} configurations tried)".format(next_configuration, self.num_configs_tried))
        return next_configuration


class ListOptimizer(object):
    '''Evaluates configurations drawn from a list.'''
    def __init__(self, configuration_list):
        self.evaluation_results = []
        self.configuration_list = configuration_list

    def add_results(self, evaluation_results):
        if evaluation_results.configuration == {}:
            with open("data/{0}_default".format(evaluation_results.learner), 'a') as output:
                cPickle.dump(evaluation_results.scores, output)
        else:
            with open("data/{0}_evals".format(evaluation_results.learner), 'a') as output:
                cPickle.dump((evaluation_results.scores, evaluation_results.runtime, evaluation_results.configuration), output)

    def get_next_configuration(self):
        if len(self.configuration_list) > 0:
            print("testing one of {0} configurations".format(len(self.configuration_list)))
            return self.configuration_list.pop()


'''
class GaussianMixtureOptimizer(BaseOptimizer):
    def __init__(self, param_space):
        super(GaussianMixtureOptimizer, self).__init__(param_space)
        self.GMMs = {}
        self.prior_freq = 1.

    def add_results(self, evaluation_results):
        super(GaussianMixtureOptimizer, self).add_results(evaluation_results)
        get signature from evaluation results
        get numerical hyperparameter space from signature
        for each numerical hyperparameter:
            update distribution with observation
            try to simplify distribution

    def get_next_configuration(self):
        signature = sample_signature(self.param_space)
        numerical_params = ParameterSpace.get_numerical_space(self.param_space, signature)
        GMM = GMMs.get(str(signature))
        if GMM is None:
            self.GMMs[str(signature)] = GMM  # new GMM
            # recreate all numerical distributions as gmms
        for each numerical param:
            if np.random.rand() <= self.prior_freq:
                sample from prior
            else
                sample from learned gmm

        combined = combine signature with sampled dictionary
        return combined

'''
