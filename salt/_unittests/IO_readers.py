"""Unit tests for module :mod:`salt.IO.readers`."""

import os
import unittest
from salt.IO.readers import ArffReader
from salt.utils.debug import disable_stdout, enable_stdout


class IOTests(unittest.TestCase):
    """Tests for the input/output module."""
    def setUp(self):
        self.base_file_path = os.path.dirname(os.path.realpath(__file__)) + '/../../data/test/'
        print(self.base_file_path)
        disable_stdout()

    def tearDown(self):
        enable_stdout()

    def test_read_valid_arff_file(self):
        valid_file_path = self.base_file_path + 'valid.arff'
        reader = ArffReader(valid_file_path)
        file_read = reader.read_file()
        self.assertNotEqual(file_read, None)

    def test_read_empty_arff_file(self):
        empty_file_path = self.base_file_path + "empty.arff"
        reader = ArffReader(empty_file_path)
        file_read = reader.read_file()
        self.assertEqual(file_read, None)

    def test_read_binary_arff_file(self):
        # generated with dd status=none bs=1 count=100 if=/dev/urandom of=invalid_binary.arff
        invalid_binary_file_path = self.base_file_path + "invalid_binary.arff"
        reader = ArffReader(invalid_binary_file_path)
        reader.read_file()
        self.assertRaises(Exception)

    def test_read_nonexisting_file(self):
        nonexisting_file_path = 'this_file_should_not_exist.arff'
        reader = ArffReader(nonexisting_file_path)
        reader.read_file()
        self.assertRaises(Exception)


if __name__ == '__main__':
    unittest.main()
