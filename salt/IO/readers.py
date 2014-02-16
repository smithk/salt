"""The :mod:`salt.IO.readers` module provides classes to read different data sources."""

from scipy.io.arff.arffread import loadarff, ParseArffError
from ..utils.debug import Log
from ..data import Dataset
import numpy as np
from six import b, iteritems


def map_names(target, target_names):
    result = np.zeros(len(target))
    i = 1
    for label in target_names[1:]:
        indices = np.nonzero(target == b(label))[0]
        result[indices] = i
        i += 1
    return result


# numpy's shuffle/permutation corrupts structured arrays
def _shuffle(array_to_shuffle):
    new_order = np.random.permutation(len(array_to_shuffle))
    return array_to_shuffle[new_order]


class BaseReader(object):
    """Base class for data readers. All data readers should inherit from this class."""
    def __init__(self, file_path):
        """Construct the data reader

        :param file_path: File to be read.
        """
        # TODO: load set of files?
        self.file_path = file_path

    def read_header(self):
        """Read and return information contained in the header section of the file, if any."""
        raise NotImplementedError

    def read_file(self):
        """Read and return all information contained in the file."""
        raise NotImplementedError


class ArffReader(BaseReader):
    """Class for reading arff files."""
    def read_header(self):
        """Read and return information contained in the header section of the file, if any."""
        # TODO: Implement header reading.
        pass

    def read_file(self, is_regression=False, shuffle=True):
        """Read and return all information contained in the file."""
        dataset = None
        try:
            Log.write_start("Reading file {input_file}... ".format(input_file=self.file_path))
            file_contents = loadarff(self.file_path)  # returns tuple (data, metadata)

            file_data, file_metadata = file_contents
            file_metadata_attributes = {key.lower(): value
                                        for key, value in iteritems(file_metadata._attributes)}
            file_metadata._attributes = file_metadata_attributes
            if shuffle:
                file_data = _shuffle(file_data)

            column_names = file_data.dtype.names
            feature_names = [name for name in column_names if name != column_names[-1]]
            feature_data = file_data[feature_names].view((float, len(feature_names)))
            labels = file_data[column_names[-1]]

            dataset = Dataset(is_regression)
            dataset.data = feature_data
            dataset.DESCR = file_metadata.name
            dataset.feature_names = file_metadata.names()[:-1]
            #dataset.target_names = np.unique(labels)
            if dataset.is_regression:
                dataset.target = labels
            else:
                dataset.target_names = file_metadata._attributes['class'][1]
                dataset.target = map_names(labels, dataset.target_names)

            Log.write_end()
        except (ParseArffError, StopIteration) as invalid_file_error:
            Log.write_end(invalid_file_error)
        except IOError as file_error:
            Log.write_end(file_error)

        return dataset
