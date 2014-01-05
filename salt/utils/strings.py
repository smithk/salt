"""The :mod:`salt.utils.strings` module provides string formatting utilities."""

from six import iteritems
import time


def format_dict(dictionary, separator="\n\t"):
    """Format a dictionary to be displayed over different lines, one per item.

    :param dictionary: Dictionary to format. The values need not be strings.
    :param separator: String to use as the item separator.
    """
    return "\t" + separator.join(["{key} = {value}".format(key=key, value=str(value))
                                  for (key, value) in iteritems(dictionary)])


def now():
    return time.strftime("<%T>")
