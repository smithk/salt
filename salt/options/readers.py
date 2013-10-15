"""
The :mod:`salt.options.readers` module implements classes to read user-defined
options from different sources.
"""

import argparse
from ..learn import AVAILABLE_CLASSIFIERS


class BaseReader(object):
    """Base class for option readers. All option readers should inherit from this class."""
    def get_options(self):
        """Retrieve user options."""
        raise NotImplementedError(self)


class StringReader(BaseReader):
    """Class to read user options specified as a plain-text string (e.g. the command-line)."""
    def __init__(self, option_string):
        """Build the option reader.

        :param option_string: list of strings or one-line string containing the user options.
        """
        self.parsed_options = None
        if isinstance(option_string, str):
            option_string = option_string.split()
        self.option_string = option_string

    def get_options(self):
        """Read user options from a string and set default values for missing parameters."""
        parser = argparse.ArgumentParser(description='Suggest-a-Learner Toolbox')

        parser.add_argument('-i', '--input-file',
                            help="Input file (only arff implemented at the moment)")
        parser.add_argument('-c', '--classifier', help="Classifier",
                            choices=AVAILABLE_CLASSIFIERS.keys(),
                            default=list(AVAILABLE_CLASSIFIERS.keys())[0])
        parser.add_argument('-g', '--gui', help="Use a graphical user interface",
                            action='store_true')
        parser.add_argument('-q', '--quiet', help="Supress most of the output.",
                            action='store_true')

        args = vars(parser.parse_args(self.option_string))

        self.parsed_options = args
        return self.parsed_options
