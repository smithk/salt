'''
The :mod:`salt.utils.debug` module provides functionality to track, store, and analyze process
progress, input, output, and intermediate steps.
'''

import os
import sys
import traceback
from functools import wraps
from six import print_


def log_step(message, exception=Exception, show_traceback=False):
    '''Append to the log the execution status for any given function.

    Use this function as a decorator for other functions. For example::

        @log_step('The message you want to see in the log', exception=SomeException)
        def some_function(*params, **kwparams):
            [statements...]

    :param message: Message to output to the log.
    :param exception: Expected type of exception to report failure.
    :param show_traceback: Whether or not to include traceback details in the output.
    '''
    def wrap(func):
        '''Function wrapper.'''
        @wraps(func)
        def func_call(*args, **kwargs):
            '''Function wrapper to pass arguments.'''
            try:
                Log.write_start(message)
                result = func(*args, **kwargs)
                Log.write_end()
                return result
            except exception as ex:
                Log.write_end(ex, show_traceback)
        return func_call
    return wrap


def disable_stdout():
    '''Disable message display to the standard output.'''
    sys.stdout = open(os.devnull, 'w')


def enable_stdout():
    '''Enable message display to the standard output.'''
    sys.stdout = sys.__stdout__


class Log(object):
    '''Write entries to the debug log/screen.'''
    _COLORS = {'HEADER': '\033[95m', 'OKBLUE': '\033[94m', 'OKGREEN': '\033[92m',
               'WARNING': '\033[93m', 'FAIL': '\033[91m', 'ENDC': '\033[0m'}

    supress_output = False

    @staticmethod
    def _colortext(message, color):
        '''Wrap a string around color specifier indicators.

        Use this function to create a colored string.
        '''
        return '{start}{message}{end}'.format(start=Log._COLORS[color], message=message,
                                              end=Log._COLORS['ENDC'])

    @staticmethod
    def _ok(message):
        '''Display the [OK] message for successful process exit status.'''
        return Log._colortext(message, 'OKGREEN')

    @staticmethod
    def _err(message):
        '''Display the [FAIL] message for failed process exit status.'''
        return Log._colortext(message, 'FAIL')

    @staticmethod
    def write_color(message, color='HEADER', end='\n'):
        '''
        Write a colored message to the log.

        :param message: Message to output to the log.
        :param color: Text color to use. Choose between 'HEADER', 'OKBLUE', 'OKGREEN', \
        'WARNING', or 'FAIL'.
        :param end: Character to end the message.
        '''
        Log.write('{message}'.format(message=Log._colortext(message, color)), end=end)

    @staticmethod
    def write(message, end='\n', force_output=False):
        '''
        Write a message to the log.

        :param message: Message to output to the log.
        :param end: Character to end the message.
        '''
        if Log.supress_output and not force_output:
            return
        print_('- {message}'.format(message=message), end=end)

    @staticmethod
    def write_start(message):
        '''Write the starting part of a log entry.

        Use this method to indicate the beginning of a process that might have different
        execution statuses, or as a partial output for processes that take long to finish.

        :param message: Message to output to the log.
        '''
        if Log.supress_output:
            return
        print_('- {message}'.format(message=message).ljust(110, '.'), end='')
        sys.stdout.flush()

    @staticmethod
    def write_end(exception=None, show_traceback=False):
        '''Write the ending part of a log entry.

        Use this process to indicate the end of a process with log entries opened with
        :func:`write_start`, to output information about the exit status (OK or FAILED)

        :param exception: Exception raised, if any.
        :param show_traceback: Whether to show traceback details or not.
        '''
        if Log.supress_output:
            return
        if exception:
            print_(Log._err('[FAILED]').rjust(18, '.'))
            print_('    >  {description}'.format(description=exception.__repr__()))
            if show_traceback:
                exception_traceback = traceback.format_exc()
                print_('    >  ' + exception_traceback.replace('\n', '\n    >  '))
        else:
            print_(Log._ok('[OK]').rjust(18, '.'))
