import numpy as np
import glob
import os
import cPickle
from sklearn.cluster import k_means
from salt.learn import AVAILABLE_CLASSIFIERS
from itertools import chain
from ranker import get_global_ranking
from salt.evaluate.evaluate import get_statistics, p_stud


def list_to_tree(evaluation_list, tree_structure):
    for evaluation in evaluation_list:
        insert(evaluation, tree_structure)
    return tree_structure


def insert(evaluation, tree_structure):
    if type(tree_structure) is list:
        tree_structure.append(evaluation)
    else:
        cat_param_name, cat_values = tree_structure
        score, runtime, configuration = evaluation  # unpack
        insert(evaluation, cat_values[configuration[cat_param_name]])


def get_k_means(evaluation_list, k=10):
    if len(evaluation_list) <= k:
        return evaluation_list

    #scores = [score for score, runtime, config in evaluation_list]
    scores = np.array([score for score, config in evaluation_list])
    means, labels, error = k_means(np.atleast_2d(scores).T, k)
    selected_models = []
    for i in xrange(len(np.unique(means))):
        indices = np.argwhere(labels == i)
        best = np.argmax(scores[indices])
        for index in indices[best]:
            selected_models.append(evaluation_list[index])
        # selected_means.append(scores[indices[best]])
    return selected_models


def get_candidates(tree_structure, n_means=10):
    if type(tree_structure) is list:
        candidate_models = [config for score, runtime, config in get_k_means(tree_structure, n_means)]
    else:
        cat_param_name, cat_values = tree_structure
        candidate_models = [get_candidates(cat_value, n_means) for cat_name, cat_value in cat_values.items()]
        candidate_models = list(chain.from_iterable(candidate_models))
    return candidate_models


def get_best(tree_structure, top_n=5, n_means=10, prefix='', alpha=0.05):
    if type(tree_structure) is list:
        candidate_models = get_k_means(tree_structure, n_means)
    else:
        cat_param_name, cat_values = tree_structure
        candidate_models = [get_best(cat_value, top_n, n_means, prefix=prefix + '    ', alpha=alpha) for cat_name, cat_value in cat_values.items()]
        # TODO rank them all by the compound criterion
        candidate_models = list(chain.from_iterable(candidate_models))
    # statistical test
    #all_scores = [simulate_n_runs(model) for model in candidate_models]
    if len(candidate_models) < 2:
        return candidate_models
    all_scores = [model[0] for model in candidate_models]
    statistics = get_statistics(all_scores)
    k = len(all_scores)
    df = max(abs(len(all_scores[0]) - k), 2)
    pvalues = np.array([p_stud(q, k, df) for q in statistics])
    #selected_model_indices = np.argwhere(pvalues >= alpha)  # Discard significantly different models
    selected_model_indices = np.argwhere(pvalues >= 0.00)  # Discard significantly different models

    evaluations = [evaluate_model(candidate_models[i]) for i in selected_model_indices]
    ranking = get_global_ranking(evaluations)
    ordering = np.argsort(ranking)
    #print(prefix + str([evaluations[i]['score'] for i in ordering]))

    selected_models = [candidate_models[selected_model_indices[index]] for index in ordering[:top_n]]
    #print(prefix + cat_param_name)
    '''
    for i in xrange(min(20, len(selected_models))):
        print(prefix + str(selected_models[i]))
    '''
    return selected_models

        #return max([get_best(cat_value) for cat_name, cat_value in cat_values.items()])


def print_summary_tree(tree_structure, prefix=''):
    if type(tree_structure) is list:
        print(prefix + str(len(tree_structure)) + " items")
    else:
        cat_param_name, cat_values = tree_structure
        print(prefix + cat_param_name)
        for cat_name, cat_value in cat_values.items():
            print(prefix + "-" + cat_name)
            print_summary_tree(cat_value, prefix + '    ')


def load_list(filenames):
    evaluation_list = []
    for filename in filenames:
        evaluation_file = open(filename)
        evaluation = ''
        while evaluation is not None:
            try:
                evaluation = cPickle.load(evaluation_file)
                evaluation_list.append(evaluation)
            except EOFError:
                evaluation = None
        evaluation_file.close()

    return evaluation_list


def get_filenames(path, wildcard='*'):
    filenames = glob.glob(os.path.join(os.path.abspath(path), wildcard))
    return filenames


def get_tree(learner):
    tree = AVAILABLE_CLASSIFIERS[learner].get_default_cfg()
    #print_tree(tree, [])
    return build_tree(tree)


def build_tree(node):
    categorical_params = node.get('CategoricalParams')
    if categorical_params is not None:
        if len(categorical_params) > 0:
            categorical_name, categorical_values = categorical_params.items()[0]
            hierarchy = {}
            if categorical_values is not None:
                for category, category_params in categorical_values.items():
                    if category != 'default':
                        hierarchy[category] = build_tree(category_params)
            return (categorical_name, hierarchy)
    else:
        return []


def print_tree(node, base_numerical, prefix=''):
    categorical_params = node.get('CategoricalParams')
    numerical_params = node.get('NumericalParams', {})
    all_numerical = base_numerical + numerical_params.keys()
    if categorical_params is not None:
        if len(categorical_params) > 0:
            categorical_name, categorical_values = categorical_params.items()[0]
            if categorical_values is not None:
                for category, category_params in categorical_values.items():
                    if category != 'default':
                        print(prefix + "processing " + categorical_name + ":" + category)
                        print_tree(category_params, all_numerical, prefix + '    ')
            prefix += '    '
    else:
        print(prefix + str(all_numerical))


def simulate_n_runs(model, runs=30):
    stdev = abs(np.random.rand())
    scores = np.random.normal(model[0], stdev, runs)
    return scores


def evaluate_model(model):
    # Generate 30 scores, runtimes, other indices
    #num_evaluations = 30
    #stdev = abs(np.random.rand()) / 1000

    #scores = np.random.normal(model[0], stdev, num_evaluations)
    #runtimes = np.random.normal(5, stdev, num_evaluations)
    scores, runtime, configuration = model

    #model_evaluations = zip(scores, runtimes)
    evaluation = {'score': np.mean(scores),
                  'speed': runtime,
                  'simplicity': len(configuration),
                  'stability': np.std(scores),
                  'interpretability': 0,
                  'failurerate': 0}

    return evaluation


def dump_candidates(path, learner):
    """Read _all files, filter best n per leaf and dump them to _candidates file."""
    #path = "/home/roger/salt/test_data/ecoli/data"
    #path = "/home/roger/salt/code/data/standard_ml_sets/classification/data"
    tree_structure = get_tree(learner)
    evaluation_list = load_list(get_filenames(path, "{0}_all".format(learner)))
    tree = list_to_tree(evaluation_list, tree_structure)
    candidates = get_candidates(tree, 10)
    cPickle.dump(candidates, open(os.path.join(os.path.abspath(path), "{0}_candidates".format(learner)), 'w'))
    #print("the candidates are {0}".format(candidates))


def score_candidates(path, learner):
    """Read _evals file, build the tree, score and return."""
    #path = "/home/roger/salt/test_data/ecoli/data"
    #path = "/home/roger/salt/code/data/standard_ml_sets/classification/data"
    import warnings
    warnings.filterwarnings('ignore')
    tree_structure = get_tree(learner)
    evaluation_list = load_list(get_filenames(path, "{0}_evals".format(learner)))
    tree = list_to_tree(evaluation_list, tree_structure)
    best = get_best(tree, 10, 10, alpha=0.05)
    print("Best models: {0}".format(len(best)))
    for b in best:
        print("ranking scores: {0}".format(evaluate_model(b)))


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    path = "/home/roger/salt/code/data/standard_ml_sets/classification/data"
    learner = "KNNClassifier"
    operation = "score"
    if len(sys.argv) == 3:
        path, learner, operation = args
    print("path is '{0}', learner is {1}, operation is {2}".format(path, learner, operation))
    print("operations: getcandidates | score")
    #path = "/home/roger/salt/code/data/standard_ml_sets/classification/data"
    #learner = 'KNNClassifier'

    if operation.lower() == 'getcandidates':
        dump_candidates(path, learner)
    else:
        score_candidates(path, learner)

    #print_summary_tree(tree)

    #_, algor = tree
    #minkowskis = algor['kd_tree'][1]['uniform'][1]['minkowski']
    #manhattans = algor['kd_tree'][1]['uniform'][1]['manhattan']
    #print([x['n_neighbors'] for _, x in minkowskis])
    #print([x['n_neighbors'] for _, x in manhattans])
    #print(np.max([score for score, runtime, config in evaluation_list]))
    #print(np.max([score for score, config in evaluation_list]))
    #print(len([score for score, config in evaluation_list if score > 0.82]))

    '''
    distribution1 = [x['n_neighbors'] for _, x in minkowskis]
    scores1 = [x for x, _ in minkowskis]
    scores2 = [x for x, _ in manhattans]
    distribution2 = [x['n_neighbors'] for _, x in manhattans]
    import matplotlib.pyplot as pl
    #pl.hist(distribution1, 90)
    #pl.hist(distribution2, 90)
    pl.hist(scores1, normed=True, alpha=0.5)
    pl.hist(scores2, normed=True, alpha=0.5)
    pl.hist(np.r_[scores1, scores2], 50, normed=True, alpha=0.9)
    pl.show(block=True)
    '''
