"""Test suite for SALT."""

import unittest
from salt._unittests.options_readers import OptionsTests
from salt._unittests.IO_readers import IOTests

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(OptionsTests))
    suite.addTest(unittest.makeSuite(IOTests))
    return suite

if __name__ == '__main__':
    test_suite = suite()
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
