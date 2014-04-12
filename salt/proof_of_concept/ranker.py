import numpy as np
from collections import OrderedDict
from salt.evaluate.evaluate import get_fixed_ranking


'''
def rank(item_list, criterion, decreasing=True):
    attribute_values = [item.get(criterion) for item in item_list]
    ordering = np.argsort(attribute_values)
    if decreasing:
        ordering = ordering[::-1]
    ranking = np.argsort(ordering)
    print(ranking)

    sorted_list = [item_list[index] for index in ordering]

    return sorted_list
'''


def get_ranking(item_list, criterion, decreasing=True):
    attribute_values = [item.get(criterion) for item in item_list]
    ranking = get_fixed_ranking(attribute_values, decreasing)

    return ranking


ranking_weights = {'score': 0.5,
                   'speed': 0.25,
                   'simplicity': 0.1,
                   'stability': 0.15,
                   'interpretability': 0.0,
                   'failurerate': 0.0}
ranking_weights = OrderedDict(sorted(ranking_weights.items()))
ranking_decreasing = {'score': True,
                      'speed': False,
                      'simplicity': False,
                      'stability': False,
                      'interpretability': False,
                      'failurerate': False}
ranking_decreasing = OrderedDict(sorted(ranking_decreasing.items()))


def get_global_ranking(item_list):
    #ranking_weights = {'score': 0.3, 'speed': 0.2, 'stability': 0.5}
    #ranking_weights = OrderedDict(sorted(ranking_weights.items()))
    # Use decreasing=False if lower values are better
    #ranking_decreasing = {'score': True, 'speed': False, 'stability': False}
    #ranking_decreasing = OrderedDict(sorted(ranking_decreasing.items()))

    ranking_matrix = None
    for criterion in ranking_weights.keys():
        ranking_criterion = get_ranking(item_list, criterion, ranking_decreasing[criterion])
        if ranking_matrix is None:
            ranking_matrix = np.atleast_2d(ranking_criterion)
        else:
            ranking_matrix = np.r_[ranking_matrix, np.atleast_2d([ranking_criterion])]

    ranking = np.array([ranking_weights.values()]).dot(ranking_matrix).ravel()
    #general_ordering = np.argsort(ranking)
    #ranked_list = [item_list[order] for order in general_ordering]
    return ranking


if __name__ == '__main__':
    ranking_objects = [{'score': 1, 'speed': 1, 'stability': 8},
                       {'score': 2, 'speed': 0.5, 'stability': 4},
                       {'score': 4, 'speed': 0.5, 'stability': 5},
                       {'score': 3, 'speed': 0.2, 'stability': 10}]

    ranking_weights = {'score': 0.3, 'speed': 0.2, 'stability': 0.5}
    ranking_weights = OrderedDict(sorted(ranking_weights.items()))
    # Use decreasing=False if lower values are better
    ranking_decreasing = {'score': True, 'speed': False, 'stability': False}
    ranking_decreasing = OrderedDict(sorted(ranking_decreasing.items()))

    ranking_matrix = None
    for criterion in ranking_weights.keys():
        ranking_criterion = get_ranking(ranking_objects, criterion, ranking_decreasing[criterion])
        if ranking_matrix is None:
            ranking_matrix = np.atleast_2d(ranking_criterion)
        else:
            ranking_matrix = np.r_[ranking_matrix, [ranking_criterion]]

    ranking = np.array([ranking_weights.values()]).dot(ranking_matrix).ravel()
    general_ordering = np.argsort(ranking)
    ranked_list = [ranking_objects[order] for order in general_ordering]
    print("general rank: {0}".format(ranked_list))

    for i, criterion in enumerate(ranking_weights.keys()):
        ordering = np.argsort(ranking_matrix[i])
        ranked_list = [ranking_objects[order] for order in ordering]
        print("rank by {0}: {1}".format(criterion, ranked_list))

    print(ranking)
    print("general ranking")
    print(ranking_matrix[:, general_ordering])

    for i, criterion in enumerate(ranking_weights.keys()):
        print("ranking by {0}".format(criterion))
        ordering = np.argsort(ranking_matrix[i])
        print(ranking_matrix[:, ordering])
