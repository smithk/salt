"""Base classes for data representation."""

import numpy as np
from sklearn.datasets.base import Bunch


__all__ = ['Dataset']


class Dataset(Bunch):
    """Contains data."""

    def split(self, items, shuffle=True):
        """Split a dataset."""
        num_items = self.data.shape[0]
        indices = np.arange(num_items)
        if shuffle:
            np.random.shuffle(indices)
        if isinstance(items, float):
            items = int(np.round(num_items * items))
        indices_training = indices[:items]
        indices_testing = indices[items:]

        return (self.data[indices_training], self.data[indices_testing])

    def get_slice(self, i, num_slices):
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
