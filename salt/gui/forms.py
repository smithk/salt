"""
The :mod:`salt.gui.forms` module provides an implementation of the forms needed
for the user to interact with salt on a graphical interface.
"""

# pylint: disable=F0401,R0901,R0904,R0924
from os.path import join, dirname
from six.moves import tkinter as tk
from six.moves import tkinter_tkfiledialog as filedialog
from six import PY3
from ..utils.debug import Log
from ..utils.strings import format_dict
from ..IO.readers import ArffReader
from ..learn import AVAILABLE_CLASSIFIERS, AVAILABLE_REGRESSORS  # , DEFAULT_REGRESSORS, DEFAULT_CLASSIFIERS
from ..suggest import SuggestionTaskManager
from ..evaluate import EvaluationResults
from .controls import ToolTip
from multiprocessing import Queue
from Queue import Empty as EmptyQueue
import itertools
from ..options import Settings
if PY3:
    from tkinter import ttk
else:
    import ttk


class SaltMain(ttk.Frame):
    """Main [SALT] window."""
    def __init__(self, options, master=None):  # pylint: disable=W0231
        self.btn_action = None
        self.btn_quit = None
        self.root = tk.Tk()
        self.root.title("Suggest-a-Learner Toolbox")
        #self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.options = options
        self.file_to_open = None
        ttk.Frame.__init__(self, master)
        self.pack(fill=tk.BOTH, expand=1)
        self.message_queue = Queue()
        self.setup_gui()
        #self.message_queue = self.console.queue
        self.update_content()

    def setup_gui(self):
        """Create all user controls."""
        images_path = join(dirname(__file__), 'images')

        self.toolbar = ttk.Frame(self)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)

        open_image = tk.PhotoImage(file="{path}/open.gif".format(path=images_path))
        run_image = tk.PhotoImage(file="{path}/run3.gif".format(path=images_path))
        self.btn_open = ttk.Button(self.toolbar, text="Do SALT", image=open_image)
        self.btn_open["command"] = self.show_open_file
        self.btn_open.image = open_image
        self.btn_open.pack(side=tk.LEFT)
        ToolTip(self.btn_open, msg="Select your dataset")

        self.btn_run = ttk.Button(self.toolbar, image=run_image, state=tk.DISABLED)
        self.btn_run["text"] = "Run"
        self.btn_run["command"] = self.process_file  # self.quit
        self.btn_run.image = run_image
        self.btn_run.pack(side=tk.LEFT, padx=5)
        ToolTip(self.btn_run, msg="Start optimization")

        self.dataset_type = tk.BooleanVar()
        self.dataset_type.set(False)
        self.dataset_type_container = ttk.Frame(self.toolbar)
        self.dataset_type_container.pack(side=tk.RIGHT)
        self.btns_dataset_type_c = ttk.Radiobutton(self.dataset_type_container, text='Classification', variable=self.dataset_type, value=False)
        self.btns_dataset_type_r = ttk.Radiobutton(self.dataset_type_container, text='Regression', variable=self.dataset_type, value=True)
        self.btns_dataset_type_c.pack(side=tk.TOP)
        self.btns_dataset_type_r.pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(self)

        self.console_frame = ttk.Frame(self)  # , relief=tk.RAISED, borderwidth=2)
        self.console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1, padx=2, pady=2)

        #self.console = Console(self.console_frame, bg="black", fg="#aaa", state=tk.DISABLED, font=("Ubuntu Mono", 9))
        self.console = ttk.Tkinter.Text(self.console_frame, bg="black", fg="#aaa", font=("Ubuntu Mono", 9))
        #from tkFont import families
        #print(families())
        self.console_scroll = ttk.Scrollbar(self.console_frame)
        self.console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_scroll.config(command=self.console.yview)
        self.console.config(yscrollcommand=self.console_scroll.set)

        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Tree view example
        self.ranking_frame = ttk.Frame(self)
        self.ranking_frame.pack(fill=tk.BOTH, expand=1, padx=2, pady=2)

        self.tree_rank = ttk.Treeview(self.ranking_frame, columns=('configuration', 'score'))
        self.tree_rank.heading("#0", text="Learner")
        self.tree_rank.heading(0, text="Configuration")
        self.tree_rank.heading(1, text="Score")
        self.tree_global_rank = self.tree_rank.insert('', 1, text="Global ranking")
        self.learner_ranks = {}
        self.ranking_scroll = ttk.Scrollbar(self.ranking_frame)
        self.ranking_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.ranking_scroll.config(command=self.tree_rank.yview)
        self.tree_rank.config(yscrollcommand=self.ranking_scroll.set)
        self.tree_rank.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        #test to embed controls into a canvas for scrolling
        '''
        self.control_container = ttk.Frame(self)
        self.canvas_frame = tk.Canvas(self.control_container)
        self.canvas_scroll = ttk.Scrollbar(self.control_container)
        self.canvas_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_scroll.config(command=self.canvas_frame.yview)
        self.canvas_frame.config(yscrollcommand=self.canvas_scroll.set)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.buttonc = ttk.Button(self, text="Text")
        self.canvas_frame.create_window(0,0, window=self.buttonc, anchor=tk.N+tk.W)
        self.notebook.add(self.control_container, text="[test]")
        '''

        #test to embed matplotlib
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
        from matplotlib.figure import Figure
        import numpy as np

        example_figure = Figure(figsize=(5, 4), dpi=100)
        subplot = example_figure.add_subplot(111)
        x = np.arange(0.0, 3.0, 0.01)
        y = np.sin(2 * np.pi * x)

        subplot.plot(x, y)
        subplot.set_title('Tk embedding example')
        subplot.set_xlabel('X label')
        subplot.set_ylabel('Y label')

        self.plot_frame = ttk.Frame(self)
        self.plot = FigureCanvasTkAgg(example_figure, master=self.plot_frame)
        self.plot.show()
        self.plot.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.plot._tkcanvas.pack(fill=tk.BOTH, expand=1)
        self.plot_toolbar = NavigationToolbar2TkAgg(self.plot, self.plot_frame)
        self.plot_toolbar.update()
        self.plot_frame.pack()
        '''
        '''
        # Settings pane
        self.settings_frame = ttk.Frame(self)
        self.settings_scroll_container = ttk.Frame(self.settings_frame, borderwidth=2, relief='ridge')
        self.settings_scrollbar = ttk.Scrollbar(self.settings_scroll_container)
        self.settings_container = ttk.Frame(self.settings_scroll_container)
        self.settings_toolbar = ttk.Frame(self.settings_frame)
        self.btn_save = ttk.Button(self.settings_toolbar, text='Save settings')
        self.btn_restore = ttk.Button(self.settings_toolbar, text='Restore to saved settings')

        self.btn_save.pack(side=tk.RIGHT)
        self.btn_restore.pack(side=tk.RIGHT)
        self.settings_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.settings_scroll_container.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.settings_toolbar.pack(side=tk.BOTTOM, fill=tk.X, expand=0)
        self.settings_frame.pack(fill=tk.BOTH, expand=1)

        self.notebook.add(self.ranking_frame, text="Ranking")
        self.notebook.add(self.plot_frame, text="Plot test")
        self.notebook.add(self.console_frame, text="Console")
        self.notebook.add(self.settings_frame, text="Settings")
        self.notebook.pack(fill=tk.BOTH, expand=1)

        ttk.Style().theme_use('clam')

    def show_open_file(self):  # pylint: disable=R0201
        """Shows the Open file dialog."""
        self.file_to_open = filedialog.askopenfilename(filetypes={("Arff files", "*.arff")})
        if self.file_to_open:
            self.btn_run.configure(state=tk.NORMAL)
            #print("open file {thefile}".format(thefile=self.file_to_open))
            self.message_queue.put("open file {thefile}\n".format(thefile=self.file_to_open))

    def load_dataset(self, is_regression):
        if self.file_to_open:
            arff_reader = ArffReader(self.file_to_open)
            dataset = arff_reader.read_file(is_regression)

            return dataset

    def create_default_parameters(self, learners):
        parameter_dict = {learner.__name__: learner.create_default_params() for learner in learners}
        return parameter_dict

    def load_settings(self, options):
        settings = Settings.load_or_create_config()
        return settings

    def report_results(self, ranking):
        string_template = "\tscore: {score:.8f} learner: [{learner}] with parameters {parameters}"

        all_results = sorted(itertools.chain.from_iterable(ranking.values()), reverse=True)
        print("\nGlobal ranking:")
        for result in all_results:
            print(string_template.format(score=result.metrics.score,
                                         learner=result.learner,
                                         parameters=result.parameters))

        print("\nLearner-wise ranking:")
        for learner in ranking.keys():
            print("- {learner}:".format(learner=learner))
            for result in ranking[learner]:
                print(string_template.format(score=result.metrics.score,
                                             learner=learner,
                                             parameters=result.parameters))

    def create_parameter_space(self, learners, settings, optimizer):
        parameter_dict = {learner.__name__: learner.create_parameter_space(settings, optimizer) for learner in learners}
        return parameter_dict

    def process_file(self):
        self.reset_tree()
        Log.write("Running [SALT] in gui mode with options\n{options}".format(options=format_dict(self.options)))

        # === Dataset ===
        dataset = self.load_dataset(self.dataset_type.get())
        if not dataset:
            return  # break?
        dataset.initialize()

        settings = self.load_settings(None)

        # === Learners ===
        classifier_settings = settings.get('Classifiers')
        regressor_settings = settings.get('Regressors')
        default_classifiers = None
        default_regressors = None
        optimizer = 'KDEOptimizer'
        if classifier_settings:
            default_classifiers = [AVAILABLE_CLASSIFIERS[key] for key in classifier_settings
                                   if classifier_settings[key][optimizer].get('enabled', False)]
        if regressor_settings:
            default_regressors = [AVAILABLE_REGRESSORS[key] for key in regressor_settings
                                  if regressor_settings[key][optimizer].get('enabled', False)]
        #import sys
        # sys.exit(0)
        learners = default_regressors if dataset.is_regression else default_classifiers
        # for testing:
        #learners = DEFAULT_REGRESSORS if dataset.is_regression else DEFAULT_CLASSIFIERS
        if not learners or len(learners) == 0:
            return  # break?

        # === Metrics ===
        metrics = []
        if metrics is None:  # or len(metrics) == 0:
            print("No metrics")
            return  # break?

        #parameters = self.create_default_parameters(learners)
        parameter_space = self.create_parameter_space(learners, settings, optimizer)

        suggestion_task_manager = SuggestionTaskManager(dataset, learners, parameter_space, metrics,
                                                        time=settings['Global'].as_int('timeout'), report_exit_caller=self.report_results, console_queue=self.message_queue)
        suggestion_task_manager.run_tasks(wait=False)

    def reset_tree(self):
        for learner in self.learner_ranks.values():
            self.tree_rank.delete(learner)
        self.tree_rank.delete(self.tree_global_rank)
        self.tree_global_rank = self.tree_rank.insert('', 1, text="Global ranking")
        self.learner_ranks = {}

    '''
    def update_tree(self, optimization_results, top_n=5):
        for learner, results in optimization_results.iteritems():
            if learner not in self.learner_ranks:
                self.learner_ranks[learner] = self.tree_rank.insert('', tk.END, text=learner)
            for result in results:
                self.tree_rank.insert(self.learner_ranks[learner], tk.END, text=learner,
                                      values=(result.parameters, result.metrics.score))
    '''
    def add_result(self, eval_result, top_n=5):
        if eval_result.learner not in self.learner_ranks:
            self.learner_ranks[eval_result.learner] = self.tree_rank.insert('', tk.END, text=eval_result.learner)
        learner_node = self.learner_ranks[eval_result.learner]
        learner_results = self.tree_rank.get_children(learner_node)
        i = 0
        for result in learner_results:
            score = float(self.tree_rank.item(result).values()[2][1])
            if score <= eval_result.metrics.score:
                #if i >= top_n:
                #    self.tree_rank.detach(result)
                #    print(result)
                #    break
                break
            i += 1
        if i < top_n or True:
           inserted_result = self.tree_rank.insert(learner_node, i, text=eval_result.learner, values=(eval_result.parameters, eval_result.metrics.score))
        global_results = self.tree_rank.get_children(self.tree_global_rank)
        i = 0
        for result in global_results:
            score = float(self.tree_rank.item(result).values()[2][1])
            if score < eval_result.metrics.score:
                break
            i += 1
        self.tree_rank.insert(self.tree_global_rank, i, text=eval_result.learner, values=(eval_result.parameters, eval_result.metrics.score))

    def update_content(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                if message is None:
                    self.console.config(state=tk.NORMAL)
                    self.console.delete(1.0, ttk.Tkinter.END)
                else:
                    if type(message) is str:
                        self.console.config(state=tk.NORMAL)
                        self.console.insert(ttk.Tkinter.END, message)
                    elif type(message) is EvaluationResults:
                        self.add_result(message)
                        #self.tree_rank.insert(self.tree_global_rank, tk.END, text=message.learner, values=(message.parameters, message.metrics.score))
                        #if message.learner not in self.learner_ranks:
                        #    self.learner_ranks[message.learner] = self.tree_rank.insert('', tk.END, text=message.learner)
                        #self.tree_rank.insert(self.learner_ranks[message.learner], tk.END, text=message.learner, values=(message.parameters, message.metrics.score))
                    elif type(message) is dict:
                        pass
                        #self.update_tree(message)
                        #self.message_queue.put(str(message))
                    else:
                        print("HHHHHHHHHHHHHHHHH {0}".format(type(message)))
                self.console.see(ttk.Tkinter.END)
                self.console.update_idletasks()
                #self.config(state=tk.DISABLED)
        except EmptyQueue:
            pass
        self.after(100, self.update_content)

    def show(self):
        """Display the main window."""
        Log.write("Loading GUI with options \n{options}".format(
            options=format_dict(self.options)))

        self.root.mainloop()
        #self.root.destroy()
