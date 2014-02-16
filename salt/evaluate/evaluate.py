"""The :mod:`salt.evaluate.base` module provides general classes for learner evaluation."""
from collections import Mapping
from six import iteritems
import numpy as np
from scipy import stats
from salt.utils.qsturng import qsturng as q_stud, psturng as p_stud
from warnings import warn


class EvaluationResults(object):
    def __init__(self, learner, parameters, metrics):
        self.learner = learner
        self.parameters = parameters
        self.metrics = metrics

    def __lt__(self, other):
        return self.metrics.score < other.metrics.score


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


def test_normally_distributed(datapoints, alpha=0.05):
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


def estimated_deviation(datapoints):
    model_performances = datapoints
    n = len(model_performances[0])
    k = len(model_performances)
    print(k, n)
    S_vne = np.sum([np.sum((model_performances[i] - np.mean(model_performances[i])) ** 2) for i in range(k)]) / (1. * n * k - k)

    return S_vne * np.sqrt(2. / n)


def pairwise_tukey(model_performance_a, model_performance_b, SE):
    qs = np.abs(np.mean(model_performance_a) - np.mean(model_performance_b)) / SE
    return qs


def tukey_old(datapoints):
    SE = estimated_deviation(datapoints)
    df = np.min([len(model_performance) for model_performance in datapoints]) - 1
    k = len(datapoints)
    for i, model_performance in enumerate(datapoints):
        for j in xrange(i, k):
            hsd = pairwise_tukey(model_performance, datapoints[j], SE)
            if hsd > q_stud(0.95, k, df):
                print("significantly different {0:00.0}".format(p_stud(hsd, k, df)))


def tukey(datapoints):
    SE = standard_error(datapoints)
    #df = np.min([len(model_performance) for model_performance in datapoints]) - 1
    #k = len(datapoints)
    best = np.argsort([np.mean(model_performance) for model_performance in datapoints])[-1]
    qs_list = []
    for i, model_performance in enumerate(datapoints):
        #if i == best:
        #    continue
        qs = pairwise_tukey(model_performance, datapoints[best], SE)
        qs_list.append(qs)
        #if hsd < q_stud(0.95, k, df):
        #    same.append(i)
    return np.array(qs_list)


def pairwise_nemenyi(rank_1, rank_2, k, N):
    nem = np.abs(rank_2 - rank_1) / np.sqrt(k*(k+1.) /(6. * N))
    print(rank_1, rank_2, nem)
    return nem


def get_ranking(datapoints):
    means = [np.mean(model) for model in datapoints]
    order = np.argsort(means)[::-1]
    ranking = order.argsort() + 1.0
    # TODO Do this properly
    return ranking
    """
    print(means, ranking)
    sorted_means = means[order]
    # detect ties:
    last_mean = None
    num_tied = 0
    for i, mean in enumerate(sorted_means):
        if mean == last_mean:
            num_tied += 1
        elif last_mean is not None:
            for j in range(num_tied):
                ranking[ranking == i-j] = ranking[i]
            ranking
        else:
            pass
    """


def get_ranking_(datapoints):
    means = [np.mean(model) for model in datapoints]
    order_1 = np.argsort(means)[::-1]
    rank_0 = order_1.argsort()
    print("means", means)
    print("rank_0", rank_0)
    order = np.argsort(means)
    #means = np.sort(means)
    rank_0 = np.float_(rank_0)
    order_1 = np.float_(order_1)
    unique_means, unique_indices = np.unique(means, return_index=True)
    print(unique_means, unique_indices)
    return
    unique_indices = np.hstack((unique_indices, len(means)))
    num_repeated = unique_indices[1:] - unique_indices[:-1]
    rankings_to_correct = np.argwhere(num_repeated > 1).ravel()
    for ranking_to_correct in rankings_to_correct:
        correction = num_repeated[ranking_to_correct]
        nums_to_correct = order_1[unique_indices[ranking_to_correct]:unique_indices[ranking_to_correct + 1]]
        order_1[unique_indices[ranking_to_correct]:unique_indices[ranking_to_correct + 1]] = np.mean(nums_to_correct)
    print(order_1[rank_0])
    return rank_0

    """
    if len(means) != len(np.unique(means)):
        count_equal = 0
        temp_val = -10001
        for i in range(len(means)):
            if temp>0
    """


def nemenyi(datapoints):
    warn("using Tukey test")
    return tukey(datapoints)
    """
    ranking = get_ranking(datapoints)
    num_models = len(datapoints)
    N = np.sum(len(model) for model in datapoints)
    nem_values = np.array([pairwise_nemenyi(ranking[0], ranking[j], num_models, N)
                           for j in range(1, num_models)])
    return nem_values
    """


def get_statistics(datapoints):
    if type(datapoints) is list:
        datapoints = np.array(datapoints)
    print("Performing analysis over {0} sets".format(len(datapoints)))
    is_normally_distributed = test_normally_distributed(datapoints)
    results = None
    if is_normally_distributed:
        are_variances_homogeneous = test_homoscedasticity(datapoints)
        if are_variances_homogeneous:
            print("using anova")
            significant_diffs = test_anova(datapoints)
            if significant_diffs:
                print("Anova found significant differences between models")
                tukey_results = tukey(datapoints)
                df = np.min([len(model_performance) for model_performance in datapoints]) - 1
                k = len(datapoints)
                print(tukey_results < q_stud(0.95, k, df))
                results = tukey_results
            else:
                print("Anova did not find significant differences between models. Tukey anyway")
                tukey_results = tukey(datapoints)
                df = np.min([len(model_performance) for model_performance in datapoints]) - 1
                k = len(datapoints)
                print(tukey_results < q_stud(0.95, k, df))
                results = tukey_results
    if results is None:  # Couldn't use ANOVA
        # Use non-parametric
        significant_diffs = test_kruskal_wallis(datapoints)
        if significant_diffs:
            nemenyi_results = nemenyi(datapoints)
            results = nemenyi_results
        else:
            nemenyi_results = nemenyi(datapoints)
            results = nemenyi_results
            #results = np.zeros(len(datapoints))
    return results

"""

def get_best(datapoints):
    # Decide the group of best sets (highest mean) that are not significantly
    # different to each other
    if type(datapoints) is list:
        datapoints = np.array(datapoints)
    print("Performing analysis over {0} sets".format(len(datapoints)))
    is_normally_distributed = test_normally_distributed(datapoints)
    if is_normally_distributed:
        are_variances_homogeneous = test_homoscedasticity(datapoints)
        if are_variances_homogeneous:
            print("using anova")
            results = test_anova(datapoints)
            if results:
                print(tukey(datapoints))
        else:
            print("using kruskall-wallis: variances not homogeneous")
            results = test_kruskal_wallis(datapoints)
    else:
        print("using kruskall-wallis: not normally distributed")
        results = test_kruskal_wallis(datapoints)
    return results
"""


def spearman(dataset1, dataset2):
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

if __name__ == '__main__':
    datapoints = {'model1': np.random.normal(0.9, 0.1, 30),
                  'model2': np.random.normal(.7, 0.1, 30),
                  'model3': np.random.normal(.3, 0.1, 30),
                  'model4': np.random.normal(.9, .5, 30)}
    #print(spearman(datapoints['model1'], datapoints['model2']))
    #results = get_statistics(datapoints.values())
    #print(datapoints.keys())
    #print("means: {0}\nvariances: {1}".format([np.mean(evaluation) for evaluation in datapoints.values()],
    #                                          [np.var(evaluation) for evaluation in datapoints.values()]))
    #print(results)
    k = len(datapoints.values())
    v = np.min([len(a) for a in datapoints.values()]) - 1
    print([p_stud(t, k, v) for t in tukey(datapoints.values())])
    print(datapoints.keys())
    #print(get_ranking([[1,2,3], [4,5,6], [1,2,3], [1,1,1,1,1]]))
    #2, 5, 2, 1
    #2.5 4 2.5 1

    #if results:
    #    print("mean differences are not significant")
    #else:
    #    print("mean differences are significant")
