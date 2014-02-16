"""Base classes for data representation."""

import numpy as np
from sklearn.datasets.base import Bunch


__all__ = ['Dataset']


class Dataset(Bunch):
    """Contains data."""
    def __init__(self, is_regression, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.is_regression = is_regression
        self.repetitions = None
        self.folds = None

    def initialize(self, num_reps, num_folds):
        indices = np.arange(len(self.target))
        self.index_list = np.array([None] * num_reps)
        for repetition in xrange(num_reps):
            indices = np.random.permutation(indices)
            self.index_list[repetition] = self.distribute(num_folds, indices)
        self.folds = len(self.index_list[0])
        if self.folds < num_folds:
            print("One of the classes in the training set contains fewer examples than "
                  "the requested number of folds ({0}). "
                  "Using {1} folds instead".format(num_folds, self.folds))

        self.repetitions = num_reps

    def split(self, items, shuffle=True, create_obj=False):
        """Split a dataset."""
        num_items = self.data.shape[0]
        indices = np.arange(num_items)
        if shuffle:
            np.random.shuffle(indices)
        if isinstance(items, float):
            items = int(np.round(num_items * items))
        indices_training = indices[:items]
        indices_testing = indices[items:]

        if create_obj:
            train_set = Dataset(is_regression=self.is_regression)
            train_set.update()
            train_set['DESCR'] = self.get('DESCR')
            train_set.data = self.data[indices_training]
            train_set.target = self.target[indices_training]
            train_set.target_names = self.target_names
            test_set = Dataset(is_regression=self.is_regression)
            test_set['DESCR'] = self.get('DESCR')
            test_set.update()
            test_set.data = self.data[indices_testing]
            test_set.target = self.target[indices_testing]
            test_set.target_names = self.target_names
            return train_set, test_set
        else:
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
        slice_dataset['DESCR'] = self.get('DESCR')
        slice_dataset['data'] = self.data[slice_range]
        slice_dataset['target'] = self.target[slice_range]

        the_rest_dataset = Dataset()
        the_rest_dataset['DESCR'] = self.get('DESCR')
        the_rest_dataset.update(self)
        the_rest_dataset['data'] = self.data[data_indices]
        the_rest_dataset['target'] = self.target[data_indices]
        return slice_dataset, the_rest_dataset

    def get_fold_data(self, repetition, fold):
        folds = self.folds
        #if self.index_list[repetition] is None:
        #    self.index_list = self.distribute(folds)

        testing_set = Dataset(is_regression=self.is_regression)
        testing_set.update(self)
        testing_set.data = self.data[self.index_list[repetition][fold]]
        testing_set.target = self.target[self.index_list[repetition][fold]]

        training_set_indices = np.hstack([self.index_list[repetition][i]
                                          for i in xrange(len(self.index_list[repetition]))
                                          if i != fold])
        training_set = Dataset(is_regression=self.is_regression)
        training_set.update(self)
        training_set.data = self.data[training_set_indices]
        training_set.target = self.target[training_set_indices]

        return (testing_set, training_set)

    def get_target(self, repetition, fold_num=None):
        if repetition is None:
            return self.target
        if fold_num is None:
            target_indices = np.hstack(np.hstack(self.index_list[repetition]))
        else:
            target_indices = self.index_list[repetition][fold_num]
        return self.target[target_indices]

    def distribute(self, folds, indices=None):
        if indices is None:
            indices = np.arange(len(self.target))
        index_list = []
        if self.is_regression:
            index_list = np.split(indices, np.rint(np.linspace(0, len(indices), folds + 1)))[1:-1]
        else:
            # Calculates the indices for data points, so that each cross-validation
            # fold has at least one data point of each class
            subsets = []
            for i in range(len(np.unique(self.target))):
                subset = indices[self.target[indices] == i]
                subsets.append(subset)
            ####print(subsets)

            lengths = np.array(map(len, subsets))
            min_folds = np.min(lengths)
            if min_folds < folds:
                folds = min_folds

            # initial_indices contains one index pointing to an instance of each
            # class, for each fold
            #initial_partitions = [np.split(np.random.permutation(subset), [folds]) for subset in subsets]
            initial_partitions = [np.split(subset, [folds]) for subset in subsets]
            initial_indices = map(list, zip(*map(lambda x: x[0].astype(int), initial_partitions)))
            ####print("initial indices", initial_indices)
            remaining_indices = np.random.permutation(np.hstack([x[1].astype(int) for x in initial_partitions]))
            #remaining_indices = np.hstack([x[1].astype(int) for x in initial_partitions])
            ####print("remaining_indices", remaining_indices)
            remaining_fold_indices = np.split(remaining_indices, np.rint(np.linspace(0, len(remaining_indices), folds, endpoint=False))[1:].astype(int))

            index_list = [np.hstack([initial_indices[i], remaining_fold_indices[i]]).astype(int) for i in range(folds)]

        return index_list

if __name__ == '__main__':
    from os import system
    from random import seed
    system('clear')
    seed(10)
    np.random.seed(12)
    from salt.IO.readers import ArffReader
    arff_reader = ArffReader('/tmp/toy.arff')
    dataset = arff_reader.read_file(False, shuffle=False)
    train, test = dataset.split(0.8, shuffle=False, create_obj=True)
    train.initialize(2, 3)
    fold_test, fold_train = train.get_fold_data(0,0)
    print(fold_test.target)
    fold_test, fold_train = train.get_fold_data(0,1)
    print(fold_test.target)
    fold_test, fold_train = train.get_fold_data(0,2)
    print(fold_test.target)
    fold_test, fold_train = train.get_fold_data(1,0)
    print(fold_test.target)
    fold_test, fold_train = train.get_fold_data(1,1)
    print(fold_test.target)
    fold_test, fold_train = train.get_fold_data(1,2)
    print(fold_test.target)
    a = train.get_target(0)
    print(a)
    #print(test.data[:,0])
    #indices = np.arange(len(train.target))
    #xxx = train.distribute(2, indices)
    #print(xxx)
    #np.random.shuffle(indices)
    #xxx = train.distribute(2, indices)
    #print(xxx)
