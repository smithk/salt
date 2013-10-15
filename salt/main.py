"""Entry point for GUI and command line interface."""

import sys as _sys
import itertools
import numpy as np
from .IO.readers import ArffReader
from .utils.debug import log_step, Log
from .utils.strings import format_dict
from .options.readers import StringReader
from .learn import DEFAULT_LEARNERS
from .gui.forms import SaltMain
from .suggest import SuggestionTaskManager

suggestion_task_manager = None


@log_step("Checking the environment")
def check_environment():
    """Check the prerequisites and versions currently installed.

    :returns: Dictionary with prerequisites and versions.
    """
    from numpy import __version__ as numpy_version
    from scipy import __version__ as scipy_version
    from matplotlib import __version__ as matplotlib_version
    from sklearn import __version__ as sklearn_version
    from six.moves import tkinter  # pylint: disable=F0401
    tkinter_version = tkinter.TkVersion
    return {'numpy': numpy_version,
            'scipy': scipy_version,
            'matplotlib': matplotlib_version,
            'sklearn': sklearn_version,
            'tkinter': tkinter_version}


def load_dataset(file_path):
    if file_path:
        arff_reader = ArffReader(file_path)
        dataset = arff_reader.read_file()

        return dataset


def create_default_parameters(learners):
    parameter_dict = {learner.__name__: learner.create_default_params() for learner in learners}
    return parameter_dict


def run_cmdline(options):
    """Run [SALT] in command-line mode.

    :param options: user-specified configuration.
    """
    Log.write("Running [SALT] in command-line mode with options\n{options}".format(
        options=format_dict(options)))

    # === Dataset ===
    input_file = options['input_file']
    dataset = load_dataset(input_file)
    if not dataset:
        return  # break?

    # === Learners ===
    learners = DEFAULT_LEARNERS  # TODO: Load learners from user options.
    if not learners or len(learners) == 0:
        return  # break?

    # === Metrics ===
    metrics = []
    if metrics is None:  # or len(metrics) == 0:
        print("No metrics")
        return  # break?

    parameters = create_default_parameters(learners)

    suggestion_task_manager = SuggestionTaskManager(dataset, learners, parameters, metrics,
                                                    time=10, report_exit_caller=report_results)
    Log.write_color("\n========================= SENDING JOBS =========================", 'OKGREEN')
    suggestion_task_manager.run_tasks()


def report_results(ranking):
    # TODO: TOP should be parameter
    TOP = 5
    string_template = "\tscore: {score:.8f} learner: [{learner}] with parameters {parameters}"

    all_results = sorted(itertools.chain.from_iterable(ranking.values()), reverse=True)
    Log.write_color("\n=========================== RESULTS ===========================", 'OKGREEN')
    print("Global ranking: (top {0})".format(TOP))
    for result in all_results[:TOP]:
        print(string_template.format(score=result.metrics.score,
                                     learner=result.learner,
                                     parameters=result.parameters))

    print("\nLearner-wise ranking: (top {0} per learner)".format(TOP))
    for learner in ranking.keys():
        print("- {learner}:".format(learner=learner))
        learner_results = sorted(ranking[learner], reverse=True)
        for result in learner_results[:TOP]:
            print(string_template.format(score=result.metrics.score,
                                         learner=learner,
                                         parameters=result.parameters))


def run_gui(options):
    """Configure and display the graphical user interface.

    :param options: user-specified configuration.
    """
    gui = SaltMain(options)
    gui.show()


def main():
    """Entry point for command line execution."""
    args = _sys.argv[1:]
    option_reader = StringReader(args)  # Parse command-line.
    options = option_reader.get_options()

    if options['quiet']:
        Log.supress_output = True
    installed_modules = check_environment()
    Log.write("Installed dependencies:\n{installed_modules}".format(
        installed_modules=format_dict(installed_modules, separator='\t')))

    np.random.seed(173205)
    if options:
        if options['gui']:
            run_gui(options)
        else:
            run_cmdline(options)
