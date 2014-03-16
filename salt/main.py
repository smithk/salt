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
from .learn import AVAILABLE_CLASSIFIERS, AVAILABLE_REGRESSORS, create_parameter_space
from .gui.forms import SaltMain
from .suggest import SuggestionTaskManager
import setproctitle

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


def create_default_parameters(learners):
    parameter_dict = {learner.__name__: learner.create_default_params() for learner in learners}
    return parameter_dict


def load_settings(options):
    settings = Settings.load_or_create_config()
    return settings


def f(x):
    import time
    from datetime import datetime
    time_start = datetime.now()
    time.sleep(4)
    time_end = datetime.now()
    return time_start, time_end


def callback(x):
    print(x)
    return x


def _run_cmdline(cmdline_options):
    # ==== multiprocessing test. Delete ====

    from multiprocessing import Pool
    p = Pool(10)
    print('asdfasdf')

    tasks = []
    for i in xrange(12):
        task = p.apply_async(f, (1,), callback=callback)
        print(type(task))
        tasks.append(task)
    from datetime import datetime
    print(datetime.now())
    import time
    time.sleep(9)
    p.terminate()
    for task in tasks:
        print(task.ready())
    p.close()
    p.join()

    # ==== end of multiprocessing test ====


def run_cmdline(cmdline_options):
    """Run [SALT] in command-line mode.

    :param options: user-specified configuration.
    """
    Log.write("Running [SALT] in command-line mode with options\n{options}".format(
        options=format_dict(cmdline_options)))

    # === Settings read from file ===
    settings = load_settings(cmdline_options)
    holdout = settings['Global'].as_float('holdout')
    crossvalidation = settings['Global'].get('crossvalidation', '10')
    local_cores = settings['Global'].as_int('localcores')
    node_path_list = settings['Global'].get('nodes', ['localhost'])
    timeout = settings['Global'].as_int('timeout')
    max_jobs = settings['Global'].as_int('maxprocesses')
    classifier_settings = settings.get('Classifiers')
    regressor_settings = settings.get('Regressors')
    ip_addr = settings['Global'].get('ip_addr')

    xval_repetitions, xval_folds = Settings.get_crossvalidation(crossvalidation)
    nodes = Settings.read_nodes(node_path_list)

    # === Dataset ===
    dataset_path = cmdline_options['input_file']
    is_regression = cmdline_options['regression']
    optimizer = cmdline_options['optimizer']
    learners = cmdline_options['learners']

    try:
        console_height, console_width = _os.popen('stty size', 'r').read().split()
        np.set_printoptions(linewidth=int(console_width) - 5)
    except:
        pass

    # === Learners ===
    classifier_settings = settings.get('Classifiers')
    regressor_settings = settings.get('Regressors')

    default_classifiers = None
    default_regressors = None
    if classifier_settings:
        default_classifiers = [classifier for classifier in classifier_settings
                               if classifier_settings[classifier].get('enabled')
                               in ('True', '$True')]
    if regressor_settings:
        default_regressors = [regressor for regressor in regressor_settings
                              if regressor_settings[regressor].get('enabled')
                              in ('True', '$True')]
    if is_regression:
        learner_list = AVAILABLE_REGRESSORS.keys()
        default_learners = default_regressors
    else:
        learner_list = AVAILABLE_CLASSIFIERS.keys()
        default_learners = default_classifiers

    learners = [learner for learner in learners if learner in learner_list] \
        if learners else default_learners
    if len(learners) == 0:
        print("Invalid learner set")
        return

    parameter_space = create_parameter_space(learners, settings,
                                             AVAILABLE_REGRESSORS if is_regression
                                             else AVAILABLE_CLASSIFIERS)

    train_set, hold_out_set = ArffReader.load_dataset(dataset_path, is_regression, holdout)
    if not train_set:
        return  # break?

    train_set.initialize(xval_repetitions, xval_folds)

    suggestion_task_manager = SuggestionTaskManager(train_set, learners, parameter_space,
                                                    metrics=[], time=timeout,
                                                    report_exit_caller=report_results,
                                                    console_queue=None,
                                                    command_queue=None,
                                                    local_cores=local_cores,
                                                    node_list=nodes, optimizer=optimizer,
                                                    max_jobs=max_jobs,
                                                    ip_addr=ip_addr, distributed=False)
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
    print("Main PID={0}".format(_os.getpid()))
    installed_modules = check_environment()
    Log.write("Installed dependencies:\n{installed_modules}".format(
        installed_modules=format_dict(installed_modules, separator='\t')))

    np.random.seed(173205)
    if options:
        if options['gui']:
            run_gui(options)
        else:
            run_cmdline(options)
