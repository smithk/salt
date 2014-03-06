"""The :mod:`salt.optimize.base` module provides classes for optimization."""

from bisect import insort
from numpy.random import shuffle
import numpy as np
from six import iteritems
from six.moves import range
from ..parameters.param import Uniform, LogUniform, Normal, LogNormal, ParameterSpace
from ..evaluate.evaluate import ResultSet
from copy import deepcopy


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
        self.configurations = [{}]
        numerical_params = param_space.numerical_params
        self.numerical_params_exist = len(numerical_params) > 0
        # TODO Do this correctly

    def configurations_exhausted(self):
        print("TESTING IF THE PARAMETER SPACE HAS BEEN EXHAUSTED")
        return not self.numerical_params_exist

    def get_next_configuration(self):
        next_configuration = self.param_space.sample_configuration()
        while next_configuration in self.configurations:
            #if self.configurations_exhausted():
            #    next_configuration = None
            #    break
            next_configuration = self.param_space.sample_configuration()
        self.configurations.append(next_configuration)
        return next_configuration


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
        #print("default settings are: ", default)
        self.initial_configs = []  # [default]  # Try first with default configuration
        numerical_params = param_space.numerical_params
        self.numerical_params_exist = len(numerical_params) > 0
        # TODO Do this correctly
        #self.best_result_groups = []
        #self.hypercube_results = {}

        self.hypercube_scores = {}
        self.num_configs_tried = 0
        self.mean_sorted_score_lists = []  # Keep score lists (ResultSet) sorted by their mean (increasingly)

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
        print("created hypercube: {0}".format(hypercube))
        return hypercube

    def get_hypercube(self, configuration):
        signature = ParameterSpace.get_cat_signature(self.param_space, configuration)
        hypercube = self.hypercubes.get(str(signature))
        if hypercube is None:
            numerical_params = ParameterSpace.get_numerical_space(self.param_space, configuration)
            hypercube = ShrinkingHypercubeOptimizer.create_hypercube(numerical_params)
            self.hypercubes[str(signature)] = hypercube
        return hypercube

    def get_hypercube_scores(self, configuration):
        signature = ParameterSpace.get_cat_signature(self.param_space, configuration)
        hypercube_score_list = self.hypercube_scores.get(str(signature))
        if hypercube_score_list is None:
            hypercube_score_list = {}
            self.hypercube_scores[str(signature)] = hypercube_score_list
        return hypercube_score_list

    def get_score_list(self, configuration, hypercube_score_list):
        # Obtain the score list for a specific configuration in their
        # respective hypercube (the one matching its signature)
        score_list = hypercube_score_list.get(str(configuration))
        if score_list is None:
            score_list = ResultSet(configuration)
            hypercube_score_list[str(configuration)] = score_list
        return score_list

    def add_results(self, evaluation_results):
        # Optimizes means
        super(ShrinkingHypercubeOptimizer, self).add_results(evaluation_results)
        # Get the list of results for the current hypercube (the one that
        # matches the signature), as a dictionary {configuration: result_set}
        hypercube_score_list = self.get_hypercube_scores(evaluation_results.parameters)

        # List of scores (ResultSet) for the current configuration
        config_score_list = self.get_score_list(evaluation_results.parameters, hypercube_score_list)
        config_score_list.add(evaluation_results.metrics.score)

        last_best = None
        if len(self.mean_sorted_score_lists) > 0:
            last_best = self.mean_sorted_score_lists[-1]
        if config_score_list in self.mean_sorted_score_lists:
            self.mean_sorted_score_lists.remove(config_score_list)
        insort(self.mean_sorted_score_lists, config_score_list)
        new_best = self.mean_sorted_score_lists[-1]

        # TODO Fix bug when converting param.default to float (it can be None)
        '''
        if last_best is not None:
            print("comparing *{0} ({1} points) to {2} ({3} points)".format(last_best.mean,
                                                                           len(last_best.scores),
                                                                           config_score_list.mean,
                                                                           len(config_score_list.scores)
                                                                           )
                  )
        '''
        if new_best != last_best:
            self.expand(new_best.configuration)
            '''
            print("new best for {0} {1}: {2} ({3}, id={4}) > "
                  "{5} ({6}, id={7})".format(evaluation_results.learner,
                                             id(hypercube_score_list),
                                             new_best.mean, len(new_best.scores),
                                             id(new_best),
                                             last_best.mean if last_best else '',
                                             len(last_best.scores) if last_best else '',
                                             id(last_best) if last_best else ''
                                             )
                  )
            '''
        else:
            self.shrink(last_best.configuration)  # Shrink around same config

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
        if str(signature) in self.hypercube_scores:
            try:
                self.mean_sorted_score_lists.remove(self.hypercube_scores[str(signature)])
            except:
                pass
            for score in self.hypercube_scores[str(signature)].values():
                self.mean_sorted_score_lists.remove(score)
            del self.hypercube_scores[str(signature)]
            # but keep the references in mean_sorted_score_lists

    def shrink(self, configuration):
        self.expand(configuration, self.shrink_rate)

    def get_next_configuration(self):
        if len(self.initial_configs) > 0:
            next_configuration = self.initial_configs.pop()
        else:
            if len(self.hypercubes) > 0 and all(value <= self.hypercube_threshold for hypercube in self.hypercubes.values() for value in hypercube.values()):
                self.hypercubes = {}
                self.mean_sorted_score_lists = []
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
        print("now trying {0} ({1} configurations tried)".format(next_configuration, self.num_configs_tried))
        return next_configuration


class DefaultConfigOptimizer(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super(DefaultConfigOptimizer, self).__init__(*args, **kwargs)
        self.has_run = False
        pass

    def get_next_configuration(self):
        if self.has_run:
            next_configuration = None
        else:
            next_configuration = {}
            self.has_run = True
        return next_configuration
