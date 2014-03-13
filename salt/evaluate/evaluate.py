"""The :mod:`salt.evaluate.base` module provides general classes for learner evaluation."""
from collections import Mapping
from six import iteritems
import numpy as np
from scipy import stats
from salt.utils.qsturng import qsturng as q_stud, psturng as p_stud
from warnings import warn


class ResultSet(object):
    '''Hold the list of scores for a given configuration'''
    def __init__(self, configuration):
        self.configuration = configuration
        self.scores = []
        self.mean = None

    def add(self, new_score):
        self.scores.append(new_score)
        self.mean = np.mean(self.scores)

    def __lt__(self, other):
        return self.mean < other.mean


class EvaluationResults(object):
    def __init__(self, learner, parameters, metrics):
        self.learner = learner
        self.parameters = parameters
        self.metrics = metrics

    def __lt__(self, other):
        return self.metrics.score < other.metrics.score


class EvaluationResultSet(object):
    def __init__(self, learner, configuration, evaluations):
        self.learner = learner
        self.configuration = configuration
        self._evaluations = None
        self._mean = None
        self._set_evaluations(evaluations)

    def _set_evaluations(self, evaluations):
        scores = [evaluation.score for evaluation in evaluations]
        mean = np.mean(scores)
        self._mean = mean
        self._evaluations = evaluations

    evaluations = property(lambda self: self._evaluations, _set_evaluations)
    mean = property(lambda self: self._mean)

    def __lt__(self, other):
        return self.mean < other.mean


def standardize(datapoints, sort=False):
    if isinstance(datapoints, Mapping):
        standardized = {model_name: standardize(model_performance, sort)
                        for model_name, model_performance in iteritems(datapoints)}
    else:
        mean = np.mean(datapoints)
        stdev = np.std(datapoints)
        standardized = (datapoints - mean) / stdev
        if sort:
            standardized = np.sort(standardized)
    return standardized


def test_normality(datapoints, alpha=0.05):
    '''Apply the Kolmogorov-Smirnoff test to compare the distributions
    of the datapoints with a standarized normal distribution.'''
    all_normal = True
    for model_performance in datapoints:
        standardized_performance = standardize(model_performance)  # kstest already sorts the data
        n = len(standardized_performance)
        D, p_val = stats.kstest(standardized_performance, 'norm', N=n)
        if p_val < alpha:
            all_normal = False
            break
    return all_normal


def test_homoscedasticity(datapoints, alpha=0.05):
    T, p_val = stats.bartlett(*datapoints)
    return p_val > alpha


def test_anova(datapoints, alpha=0.05):
    F, p = stats.f_oneway(*datapoints)
    return p <= alpha


def test_kruskal_wallis(datapoints, alpha=0.05):
    H, p = stats.kruskal(*datapoints)
    return p > alpha


def standard_error(datapoints):
    n = np.min([len(datapoint_set) for datapoint_set in datapoints])
    k = len(datapoints)
    means = [np.mean(datapoint_set) for datapoint_set in datapoints]

    MSE = np.sum([np.sum((datapoint_set - means[j]) ** 2)
                  for j, datapoint_set in enumerate(datapoints)]) / (n * k - k)
    se = np.sqrt(MSE / n)
    return se


def pairwise_tukey(model_performance_a, model_performance_b, SE):
    qs = np.abs(np.mean(model_performance_a) - np.mean(model_performance_b)) / SE
    return qs


def tukey(datapoints):
    '''Compute the Tukey statistics between the different model evaluation groups.'''
    SE = standard_error(datapoints)
    best = np.argmax([np.mean(model_performance) for model_performance in datapoints])
    qs_list = [pairwise_tukey(model_performance, datapoints[best], SE) for model_performance in datapoints]
    return np.array(qs_list)


def pairwise_nemenyi(rank_1, rank_2, k, N=1):
    nem = np.abs(rank_2 - rank_1) / np.sqrt(k * (k + 1.) / (6. * N))
    return nem


def get_fixed_ranking(to_rank, decreasing=True):
    '''Compute the ranking for an array of numbers, and adjust for ties.

        Parameters:
            decreasing: True if best ranking is highest number, False otherwise.

        Return:
            Ranking for the given array. Lower number is better.

    '''
    ordering = np.argsort(to_rank)
    ranking = np.float_(ordering.argsort())
    if decreasing:
        # Reverse ranking
        ranking = len(to_rank) - 1 - ranking
    sorted_data = np.array(to_rank)[ordering]
    unique, indices = np.unique(sorted_data, return_index=True)
    _indices = np.hstack((indices, len(to_rank)))
    num_reps = _indices[1:] - _indices[:-1]
    for i in np.argwhere(num_reps > 1):
        mean = np.mean(ranking[ordering[_indices[i]:_indices[i + 1]]])
        ranking[ordering[_indices[i]:_indices[i + 1]]] = mean
    ranking = ranking + 1
    return ranking


def nemenyi(datapoints):
    '''Get the Nemenyi test values for the datapoint set.'''
    # TODO Validate (or correct) assumption that one can rank averages over all
    # datasets (See Demsar 2006, p.11)
    means = [np.mean(model) for model in datapoints]
    best = np.argmax(means)
    ranking = get_fixed_ranking(means)
    k = len(means)
    N = np.min([len(model) for model in datapoints])
    nemenyi = [pairwise_nemenyi(ranking[i], ranking[best], k, N) for i in range(len(means))]
    return nemenyi


def get_statistics(datapoints, verbose=False):
    results = None
    if type(datapoints) is list:
        datapoints = np.array(datapoints)
    is_normally_distributed = test_normality(datapoints)
    if is_normally_distributed:
        are_variances_homogeneous = test_homoscedasticity(datapoints)
        if are_variances_homogeneous:
            significant_diffs = test_anova(datapoints)
            if verbose:
                if significant_diffs:
                    print("Anova found significant differences between models.")
                else:
                    print("Anova did not find significant differences between models.")
            tukey_results = tukey(datapoints)
            results = tukey_results
    if results is None:  # Couldn't use ANOVA
        # Use non-parametric
        significant_diffs = test_kruskal_wallis(datapoints)
        if verbose:
            if significant_diffs:
                print("KW test found significant differences between models.")
            else:
                print("KW test did not find significant differences between models.")
        nemenyi_results = nemenyi(datapoints)
        results = nemenyi_results
    return results


def spearman(dataset1, dataset2):
    '''Compute the Spearman's correlation coefficient between two sets of data.'''
    mean1 = np.mean(dataset1)
    mean2 = np.mean(dataset2)
    n = len(dataset1)
    assert n == len(dataset2)

    numerator = [(dataset1[i] - mean1) * (dataset2[i] - mean2) for i in range(n)]
    denominator1 = (dataset1 - mean1) ** 2
    denominator2 = (dataset2 - mean2) ** 2
    denominator = np.sqrt(np.sum(denominator1) * np.sum(denominator2))
    if denominator != 0:
        return sum(numerator) / denominator
    else:
        return None


def format_p_value(p_value):
    p_value = np.round(p_value, 4)
    if p_value <= 0.001:
        return "(<= 0.001)"
    elif p_value >= 0.9:
        return "(>= 0.9)"
    else:
        return str(p_value)


# For testing purposes
if __name__ == '__main__':
    datapoints = {'model1': np.random.normal(0.9, 0.1, 30),
                  'model2': np.random.normal(.7, 0.1, 30),
                  'model3': np.random.normal(.3, 0.1, 30),
                  'model4': np.random.normal(.9, .5, 30)}
    print(spearman(datapoints['model1'], datapoints['model2']))
    #to_rank = np.array([7,6,2,3,5,4,5,1,1,3,7,3,4,7])
    to_rank = np.array([0.005, 0.008, 0.017, 0.033, 0.006, 0.005, 0.007, 0.000, 0.063, 0.000, 0.022, 0.047, 0.009, 0.021])
    a_ranking = get_fixed_ranking(to_rank, False)
    print(to_rank, a_ranking)
