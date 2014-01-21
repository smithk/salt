"""Entry point for GUI and command line interface."""

import sys as _sys
import os as _os
import itertools
import numpy as np
from .IO.readers import ArffReader
from .utils.debug import log_step, Log
from .utils.strings import format_dict
from .options.readers import StringReader
from .options import Settings
from .learn import AVAILABLE_CLASSIFIERS, AVAILABLE_REGRESSORS, DEFAULT_REGRESSORS, DEFAULT_CLASSIFIERS
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


def load_dataset(file_path, is_regression):
    if file_path:
        arff_reader = ArffReader(file_path)
        dataset = arff_reader.read_file(is_regression)

        return dataset


def create_default_parameters(learners):
    parameter_dict = {learner.__name__: learner.create_default_params() for learner in learners}
    return parameter_dict


def create_parameter_space(learners, settings, optimizer):
    print([type(learner) for learner in learners])
    parameter_dict = {learner.__name__: learner.create_parameter_space(settings, optimizer) for learner in learners}
    return parameter_dict


def load_settings(options):
    settings = Settings.load_or_create_config()
    return settings


def run_cmdline(options):
    """Run [SALT] in command-line mode.

    :param options: user-specified configuration.
    """
    Log.write("Running [SALT] in command-line mode with options\n{options}".format(
        options=format_dict(options)))

    settings = load_settings(options)

    # === Dataset ===
    data_path = options['input_file']
    is_regression = options['regression']
    dataset = load_dataset(data_path, is_regression)
    if not dataset:
        return  # break?
    dataset.initialize()

    console_height, console_width = _os.popen('stty size', 'r').read().split()
    np.set_printoptions(linewidth=int(console_width) - 5)

    # === Learners ===
    classifier_settings = settings.get('Classifiers')
    regressor_settings = settings.get('Regressors')
    default_classifiers = None
    default_regressors = None
    if classifier_settings:
        default_classifiers = [AVAILABLE_CLASSIFIERS[key] for key in classifier_settings
                               if classifier_settings[key].get('enabled', False)]
    if regressor_settings:
        default_regressors = [AVAILABLE_REGRESSORS[key] for key in regressor_settings
                              if regressor_settings[key].get('enabled', False)]
    import sys
    # sys.exit(0)
    learners = default_regressors if is_regression else default_classifiers
    # for testing:
    learners = DEFAULT_REGRESSORS if dataset.is_regression else DEFAULT_CLASSIFIERS
    if not learners or len(learners) == 0:
        return  # break?

    # === Metrics ===
    metrics = []
    if metrics is None:  # or len(metrics) == 0:
        print("No metrics")
        return  # break?

    #parameters = create_default_parameters(learners)
    parameter_space = create_parameter_space(learners, settings, optimizer='KDEOptimizer')

    suggestion_task_manager = SuggestionTaskManager(dataset, learners, parameter_space, metrics,
                                                    time=settings['Global'].as_int('timeout'), report_exit_caller=report_results)
    Log.write_color("\n========================= SENDING JOBS =========================", 'OKGREEN')
    try:
        suggestion_task_manager.run_tasks()
    except KeyboardInterrupt:
        print "Interrupted"


def report_results(ranking):
    # TODO: TOP should be parameter
    TOP = 5
    string_template = "    score: {score:.8f} learner: [{learner}({parameters})]"

    all_results = sorted(itertools.chain.from_iterable(ranking.values()), reverse=True)
    Log.write_color("\n=========================== RESULTS ===========================", 'OKGREEN')
    print('')
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
