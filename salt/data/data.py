"""Base classes for data representation."""

import numpy as np
from sklearn.datasets.base import Bunch


__all__ = ['Dataset']


class Dataset(Bunch):
    """Contains data."""
    def __init__(self, is_regression, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.is_regression = is_regression
        self.folds = None

    def initialize(self):
        folds = 3
        self.index_list = self.distribute(folds)
        self.folds = len(self.index_list)

    def split(self, items, shuffle=True):
        """Split a dataset."""
        num_items = self.data.shape[0]
        print(num_items, "============================")
        indices = np.arange(num_items)
        if shuffle:
            np.random.shuffle(indices)
        if isinstance(items, float):
            items = int(np.round(num_items * items))
        indices_training = indices[:items]
        indices_testing = indices[items:]

        return (self.data[indices_training], self.data[indices_testing])

    def _get_slice(self, i, num_slices):
        slice_indices = np.rint(np.linspace(0, len(self.data), num_slices + 1)).astype(int)

        slice_range = list(range(slice_indices[i - 1], slice_indices[i]))
        #slice_data = self.data[slice_range]

        data_indices = list(range(len(self.data)))
        data_indices[slice_indices[i - 1]: slice_indices[i]] = []

        #the_rest = self.data[data_indices]

        # TODO: not copy().
        slice_dataset = Dataset()
        slice_dataset.update(self)
        slice_dataset['data'] = self.data[slice_range]
        slice_dataset['target'] = self.target[slice_range]

        the_rest_dataset = Dataset()
        the_rest_dataset.update(self)
        the_rest_dataset['data'] = self.data[data_indices]
        the_rest_dataset['target'] = self.target[data_indices]
        return slice_dataset, the_rest_dataset

    def get_fold_data(self, fold):
        folds = 3
        if self.index_list is None:
            self.index_list = self.distribute(folds)

        testing_set = Dataset(is_regression=self.is_regression)
        testing_set.update(self)
        testing_set.data = self.data[self.index_list[fold]]
        testing_set.target = self.target[self.index_list[fold]]

        training_set_indices = np.hstack([self.index_list[i] for i in range(len(self.index_list)) if i != fold])
        training_set = Dataset(is_regression=self.is_regression)
        training_set.update(self)
        training_set.data = self.data[training_set_indices]
        training_set.target = self.target[training_set_indices]

        return (testing_set, training_set)

    def get_target(self):
        target_indices = np.hstack(self.index_list)
        return self.target[target_indices]

    def distribute(self, folds):
        indices = np.arange(len(self.target))
        index_list = []
        if self.is_regression:
            index_list = np.split(indices, np.rint(np.linspace(0, len(indices), folds + 1)))[1:-1]
        else:
            # Calculates the indices for data points, so that each cross-validation
            # fold has at least one data point of each class
            subsets = []
            for i in range(len(np.unique(self.target))):
                subset = indices[self.target == i]
                subsets.append(subset)

            lengths = np.array(map(len, subsets))
            min_folds = np.min(lengths)
            if min_folds < folds:
                folds = min_folds

            # initial_indices contains one index pointing to an instance of each
            # class, for each fold
            initial_partitions = [np.split(np.random.permutation(subset), [folds]) for subset in subsets]
            initial_indices = map(list, zip(*map(lambda x: x[0].astype(int), initial_partitions)))
            remaining_indices = np.random.permutation(np.hstack([x[1].astype(int) for x in initial_partitions]))
            remaining_fold_indices = np.split(remaining_indices, np.rint(np.linspace(0, len(remaining_indices), folds, endpoint=False))[1:].astype(int))

            index_list = [np.hstack([initial_indices[i], remaining_fold_indices[i]]).astype(int) for i in range(folds)]

        return index_list
