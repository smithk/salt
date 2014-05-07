'''Entry point for GUI and command line interface.'''

import sys
import os
import itertools

from .IO.readers import ArffReader
from .utils.debug import log_step, Log
from .utils.strings import format_dict
from .utils import misc
from .options.readers import StringReader
from .options import Settings
from .learn import AVAILABLE_CLASSIFIERS, AVAILABLE_REGRESSORS, create_parameter_space
from .gui.forms import SaltMain
from .suggest import SuggestionTaskManager
from .suggest.rank import get_local_maxima

rank = None


@log_step('Checking the environment')
def check_environment():
    '''Check for prerequisites and their respective installed versions.


    :returns: Dictionary with prerequisites and their versions (or 'None' for
    packages unable to load).
    '''
    try:  # Attempt to import numpy
        from numpy import __version__ as numpy_version
    except:
        numpy_version = None

    try:  # Attempt to import scipy
        from scipy import __version__ as scipy_version
    except:
        scipy_version = None

    try:  # Attempt to import matplotlib
        from matplotlib import __version__ as matplotlib_version
    except:
        matplotlib_version = None

    try:  # Attempt to import scikit-learn
        from sklearn import __version__ as sklearn_version
    except:
        sklearn_version = None

    try:  # Attempt to import tkinter
        from six.moves import tkinter
        tkinter_version = tkinter.TkVersion
    except:
        tkinter_version = None

    return {'numpy': numpy_version,
            'scipy': scipy_version,
            'matplotlib': matplotlib_version,
            'sklearn': sklearn_version,
            'tkinter': tkinter_version}


def run_gui(options):
    '''Configure and display the graphical user interface.

    :param options: user-specified configuration.
    '''
    gui = SaltMain(options)  # Main window
    gui.show()


def run_cmdline(cmdline_options):
    '''Run SALT in command-line mode.

    :param options: user-specified configuration.
    '''

    # Print loaded command line options.
    Log.write("Running [SALT] in command-line mode "
              "with options\n{0}".format(format_dict(cmdline_options)))

    # === Load command line options ===
    dataset_path = cmdline_options['input_file']
    is_regression = cmdline_options['regression']
    optimizer = cmdline_options['optimizer']
    learners = cmdline_options['learners']

    mode = cmdline_options['mode']  # optimization, model_selection, or full

    misc.setup_console()  # Adjusts matrix display properties on console (numpy).

    # === Load Settings (from salt.ini) ===
    settings = Settings.load_or_create_config()  # Creates salt.ini if needed.
    holdout = settings['Global'].as_float('holdout')  # Proportion of the dataset to hold out for model selection.
    local_cores = settings['Global'].as_int('localcores')  # Number of cores to use from the local machine.
    use_cluster = settings['Global'].get('usecluster', 'False')  # Distribute processes over the network?
    use_cluster = use_cluster.lower() == 'true'
    if not use_cluster and local_cores == 0:
        print("Invalid number of local cores.")
        return
    distributed_nodes = settings['Global'].get('nodes', ['localhost'])  # Nodes for distributed computing.
    timeout = settings['Global'].as_int('timeout')  # Time alloted for optimization.
    max_jobs = settings['Global'].as_int('maxprocesses')  # Maximum number of processes to be run at a time.
    ip_addr = settings['Global'].get('ip_addr')

    # === Load enabled learners ===

    classifier_settings = settings.get('Classifiers')
    regressor_settings = settings.get('Regressors')

    crossval_optimization = settings['Global'].get('crossvalidation_optimization', '10')
    crossval_model_selection = settings['Global'].get('crossvalidation_model_selection', '3x10')

    distributed_nodes = Settings.read_nodes(distributed_nodes)

    classifier_settings = settings.get('Classifiers')
    regressor_settings = settings.get('Regressors')

    if is_regression:
        learner_list = AVAILABLE_REGRESSORS.keys()
        default_learners = Settings.get_enabled_sections(regressor_settings)
    else:
        learner_list = AVAILABLE_CLASSIFIERS.keys()
        default_learners = Settings.get_enabled_sections(classifier_settings)

    learners = [learner for learner in learners
                if learner in learner_list] if learners else default_learners
    if len(learners) == 0:
        Log.write_color('No enabled learners were specified.', 'FAIL')
        return

    # === Optimization stage ===
    if mode in ('optimization', 'full'):
        crossval_repetitions, crossval_folds = Settings.get_crossvalidation(crossval_optimization)
        candidates = run_optimization_stage(dataset_path, settings, learners,
                                            is_regression, holdout, timeout,
                                            local_cores, crossval_repetitions,
                                            crossval_folds, optimizer,
                                            max_jobs, distributed_nodes,
                                            ip_addr, use_cluster)

    # === Model selection stage ===
    if mode in ('model_selection', 'full'):
        crossval_repetitions, crossval_folds = Settings.get_crossvalidation(crossval_model_selection)
        run_model_selection_stage(dataset_path, candidates, learners, local_cores,
                                  crossval_repetitions, crossval_folds,
                                  max_jobs, distributed_nodes,
                                  ip_addr, use_cluster)


def run_optimization_stage(dataset_path, settings, learners, is_regression, holdout,
                           timeout, local_cores, crossval_repetitions, crossval_folds,
                           optimizer, max_jobs, distributed_nodes, ip_addr, use_cluster):
        parameter_space = create_parameter_space(learners, settings,
                                                 AVAILABLE_REGRESSORS if is_regression
                                                 else AVAILABLE_CLASSIFIERS)
        # TODO Ensure that both train and holdout sets have enough examples for
        # each class.
        train_set, hold_out_set = ArffReader.load_dataset(dataset_path, is_regression, holdout)
        # Store the holdout set
        misc.store_object(hold_out_set, os.path.basename(dataset_path) + '_holdout')

        train_set.initialize(crossval_repetitions, crossval_folds)

        suggestion_task_manager = SuggestionTaskManager(train_set,
                                                        learners,
                                                        parameter_space,
                                                        metrics=[], time=timeout,
                                                        report_exit_caller=report_results,
                                                        console_queue=None,
                                                        command_queue=None,
                                                        local_cores=local_cores,
                                                        node_list=distributed_nodes, optimizer=optimizer,
                                                        max_jobs=max_jobs,
                                                        ip_addr=ip_addr, distributed=use_cluster)
        Log.write_color('\n=== SENDING JOBS FOR OPTIMIZATION ===', 'OKGREEN')
        try:
            suggestion_task_manager.run_tasks()
            #candidates = get_local_maxima()
            #return candidates
        except KeyboardInterrupt:
            print 'Interrupted'


def run_model_selection_stage(dataset_path, candidates, learners, local_cores,
                              crossval_repetitions, crossval_folds, max_jobs,
                              distributed_nodes, ip_addr, use_cluster):
    parameter_space = {learner: read_configurations(learner) for learner in learners}
    train_set = misc.load_serialized_object(os.path.basename(dataset_path) + '_holdout')

    suggestion_task_manager = SuggestionTaskManager(train_set, learners, parameter_space,
                                                    metrics=[], time=None,
                                                    report_exit_caller=report_results,
                                                    console_queue=None,
                                                    command_queue=None,
                                                    local_cores=local_cores,
                                                    node_list=distributed_nodes, optimizer='list',
                                                    max_jobs=max_jobs,
                                                    ip_addr=ip_addr, distributed=use_cluster)
    Log.write_color('\n=== SENDING JOBS FOR MODEL SELECTION ===', 'OKGREEN')
    try:
        suggestion_task_manager.run_tasks()
    except KeyboardInterrupt:
        print 'Interrupted'


def read_configurations(learner):
    try:
        configurations = misc.load_serialized_object_array('data/{0}_candidates'.format(learner))
        return configurations
    except:
        return []


def report_results(ranking, top=15):
    string_template = '    score: {score:.8f} learner: [{learner}({parameters})]'

    all_results = sorted(itertools.chain.from_iterable(ranking.values()), reverse=True)
    Log.write_color('\n=========================== RESULTS ===========================', 'OKGREEN')
    print('')
    print('Global ranking: (top {0})'.format(top))
    for result in all_results[:top]:
        print(string_template.format(score=result.mean,
                                     learner=result.learner,
                                     parameters=result.configuration))

    print('\nLearner-wise ranking: (top {0} per learner)'.format(top))
    for learner in ranking.keys():
        print('- {learner}:'.format(learner=learner))
        learner_results = sorted(ranking[learner], reverse=True)
        for result in learner_results[:top]:
            print(string_template.format(score=result.mean,
                                         learner=learner,
                                         parameters=result.configuration))

    rank = all_results


def main():
    '''Entry point for command line execution.'''
    args = sys.argv[1:]
    option_reader = StringReader(args)  # Parse command-line.
    options = option_reader.get_options()

    if options['quiet']:
        Log.supress_output = True
    # print('Main PID={0}'.format(os.getpid()))
    installed_modules = check_environment()
    Log.write('Installed dependencies:\n{modules}'.format(
        modules=format_dict(installed_modules, separator='\t')))
    if not all(installed_modules.values()):
        print("Some dependencies can not be loaded.")

    if options['gui']:
        run_gui(options)
    else:
        dataset_path = options['input_file']
        if dataset_path is None:
            print("No dataset specified. Please run salt -h for help.")
            return
        run_cmdline(options)
