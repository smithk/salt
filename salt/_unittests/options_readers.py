"""Unit tests for the :mod:`salt.options` package."""

import unittest
from salt.options.readers import StringReader
from salt.utils.debug import disable_stdout, enable_stdout


class OptionsTests(unittest.TestCase):
    def setUp(self):
        disable_stdout()

    def tearDown(self):
        enable_stdout()

    def test_cmdline_array_parse_help(self):
        cmdline = 'salt.py -h'.split()[1:]
        try:
            self.cmdline_parse(cmdline)
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    #@unittest.skip("debugging")
    def test_cmdline_string_parse_help(self):
        cmdline = '-h'
        try:
            print(self.cmdline_parse(cmdline))
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    def cmdline_parse(self, cmdline):
        string_reader = StringReader(cmdline)
        return string_reader.get_options()

if __name__ == '__main__':
    unittest.main(exit=False)
