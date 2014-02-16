"""
The :mod:`salt.gui.forms` module provides an implementation of the forms needed
for the user to interact with salt on a graphical interface.
"""

# pylint: disable=F0401,R0901,R0904,R0924
from os.path import join, dirname
from six.moves import tkinter as tk
from six.moves import tkinter_tkfiledialog as FileDialog
from six import PY3, iteritems
from ..utils.debug import Log
from ..utils.strings import format_dict
from ..IO.readers import ArffReader
from ..learn import AVAILABLE_CLASSIFIERS, AVAILABLE_REGRESSORS
from ..suggest import SuggestionTaskManager
from ..evaluate import EvaluationResults
from .controls import ToolTip
from multiprocessing import Queue
from Queue import Empty as EmptyQueue
import itertools
import numpy as np
from datetime import datetime, timedelta
from ..options import Settings
import warnings
with warnings.catch_warnings():
    # Ignore wanring about location of matplotlib config file
    warnings.simplefilter("ignore")
    import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import cm
if PY3:
    from tkinter import ttk
else:
    import ttk


class CustomNavToolbar(NavigationToolbar2TkAgg):
    def __init__(self, plot_canvas, plot_window, alpha):
        NavigationToolbar2TkAgg.__init__(self, plot_canvas, plot_window)
        self.alpha = alpha
        self.alpha_slider = self._slider(0.05, 0.9)
        self._message_label.pack(side=tk.LEFT)

    def _slider(self, lower, upper, default=0.05):
        sliderlabel = tk.Label(master=self, text=u'\u03b1: ')
        slider = tk.Scale(master=self, from_=lower, to=upper, orient=tk.HORIZONTAL,
                          resolution=0.01, bigincrement=0.1, bd=1, width=10, showvalue=default, variable=self.alpha)
        slider.pack(side=tk.RIGHT)
        sliderlabel.pack(side=tk.RIGHT)
        return slider


class SaltMain(ttk.Frame):
    """Main [SALT] window."""
    def __init__(self, options, master=None):  # pylint: disable=W0231
        self.root = tk.Tk()
        self.root.title("Suggest-a-Learner Toolbox")
        #self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.options = options
        self.file_to_open = None
        ttk.Frame.__init__(self, master)
        self.pack(fill=tk.BOTH, expand=1)
        self.message_queue = Queue()
        self.command_queue = Queue()
        self.process_state = 'STOPPED'
        self.alpha = tk.DoubleVar()
        self.alpha.trace("w", self.do_smth)
        self.setup_gui()
        self.update_content()
        self.update_clock()
        self.all_results = []
        self.dataset_name = ''

    def on_form_event(self, event):
        print(event)
        self.summary_figure.tight_layout()

    def do_smth(self, *args):
        self.display_chart(**self.params)

    def setup_gui(self):
        """Create all user controls."""
        images_path = join(dirname(__file__), 'images')

        self.toolbar = ttk.Frame(self)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        self.statusbar = ttk.Frame(self)
        self.statusbartext = ttk.Label(self.statusbar, text="[Ready]")
        self.statusbartext.pack(side=tk.LEFT, fill=tk.X)
        self.sizegrip = ttk.Sizegrip(self.statusbar)
        self.sizegrip.pack(side=tk.RIGHT)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)

        open_image = tk.PhotoImage(file=join(images_path, "open.gif"))
        self.run_image = tk.PhotoImage(file=join(images_path, "gears_run.gif"))
        self.pause_image = tk.PhotoImage(file=join(images_path, "gears_pause.gif"))
        stop_image = tk.PhotoImage(file=join(images_path, "gears_stop.gif"))

        self.btn_open = ttk.Button(self.toolbar, text="Open dataset", image=open_image)
        self.btn_open["command"] = self.show_open_file
        self.btn_open.image = open_image
        self.btn_open.pack(side=tk.LEFT)
        ToolTip(self.btn_open, msg="Select your dataset")

        self.btn_run = ttk.Button(self.toolbar, image=self.run_image, state=tk.DISABLED)
        self.btn_run.text = "Run"
        self.btn_run["command"] = self.run_command  # self.quit
        self.btn_run.image = self.run_image
        self.btn_run.pack(side=tk.LEFT, padx=5)
        ToolTip(self.btn_run, msg="Start optimization")

        self.btn_stop = ttk.Button(self.toolbar, image=stop_image, state=tk.DISABLED)
        self.btn_stop.text = "Stop"
        self.btn_stop["command"] = self.stop_processing
        self.btn_stop.image = stop_image
        self.btn_stop.pack(side=tk.LEFT)
        ToolTip(self.btn_stop, msg="Stop optimization")

        self.dataset_type = tk.BooleanVar()
        self.dataset_type.set(False)
        self.dataset_type_container = ttk.Frame(self.toolbar)
        self.dataset_type_container.pack(side=tk.RIGHT)
        self.btns_dataset_type_classif = \
            ttk.Radiobutton(self.dataset_type_container, text='Classification',
                            variable=self.dataset_type, value=False)
        self.btns_dataset_type_regr = \
            ttk.Radiobutton(self.dataset_type_container, text='Regression',
                            variable=self.dataset_type, value=True)
        self.btns_dataset_type_classif.pack(side=tk.TOP)
        self.btns_dataset_type_regr.pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(self)

        self.console_frame = ttk.Frame(self)
        self.console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1, padx=2, pady=2)

        self.console = ttk.Tkinter.Text(self.console_frame, bg="black", fg="#aaa",
                                        font=("Ubuntu Mono", 9))
        self.console_scroll = ttk.Scrollbar(self.console_frame)
        self.console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_scroll.config(command=self.console.yview)
        self.console.config(yscrollcommand=self.console_scroll.set)

        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Learner results tree view
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

        palette_size = 30
        self.palette = [cm.Spectral(i / float(palette_size), 1) for i in range(palette_size)]
        np.random.shuffle(self.palette)

        self.summary_figure = Figure(figsize=(5, 4), dpi=100)
        self.subplot = self.summary_figure.add_subplot(111)
        #self.subplot.format_coord = lambda x,y: ''
        #self.subplot.set_xlim([-0.5, 5.5])
        #self.subplot.set_autoscalex_on(True)
        self.subplot.set_ylim([0, 1])
        self.subplot.xaxis.grid(False)
        self.pvalue_axis = self.subplot.twinx()
        self.pvalue_axis.set_ylim([0, 1])
        self.pvalue_axis.xaxis.grid(False)
        #plt.setp(self.subplot.get_yticklabels(), rotation='vertical', fontsize='small')
        self.summary_plot = None
        #self.summary_plot = self.subplot.bar(np.arange(5), np.zeros(5), color='rgbcmyk')
        #x = np.arange(0.0, 3.0, 0.01)
        #y = np.sin(2 * np.pi * x)

        #self.subplot.plot(x, y)

        #self.subplot.set_title('Learner performance comparison', fontproperties=font0)
        #self.subplot.set_xlabel('Learner')
        self.subplot.set_ylabel('Score')

        self.plot_frame = ttk.Frame(self)
        self.plot = FigureCanvasTkAgg(self.summary_figure, master=self.plot_frame)
        self.plot.show()
        self.plot.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.plot._tkcanvas.pack(fill=tk.BOTH, expand=1)
        self.plot_toolbar = CustomNavToolbar(self.plot, self.plot_frame, alpha=self.alpha)  # FigureCanvasTkAgg(self.summary_figure, master=self.plot_frame)
        self.plot_toolbar.update()
        self.plot_frame.pack()
        self.summary_figure.tight_layout()

        self.random_noise = np.random.normal(0, 0.1, (1000, 20))
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
        self.notebook.add(self.plot_frame, text="Visualization")
        self.notebook.add(self.console_frame, text="Console")
        self.notebook.add(self.settings_frame, text="Settings")
        self.notebook.pack(fill=tk.BOTH, expand=1)

        ttk.Style().theme_use('clam')

    def show_open_file(self):
        """Show the 'Open file' dialog."""
        # TODO Load filetypes from available file readers.
        avail_filetypes = (("Arff files", "*.arff"),)
        self.file_to_open = FileDialog.askopenfilename(filetypes=avail_filetypes)
        if self.file_to_open is not None:
            self.btn_run.configure(state=tk.NORMAL)
            self.message_queue.put("open file {thefile}\n".format(thefile=self.file_to_open))

    def load_dataset(self, is_regression, hold_out=0.3):
        if self.file_to_open:
            arff_reader = ArffReader(self.file_to_open)
            dataset = arff_reader.read_file(is_regression)
            train_set, test_set = dataset.split(1 - hold_out, create_obj=True)

            return train_set, test_set

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
        parameter_dict = {learner.__name__: learner.create_param_space(settings) for learner in learners}
        return parameter_dict

    def stop_processing(self):
        self.command_queue.put("STOP")
        self.process_state = 'STOPPED'
        self.btn_open.configure(state=tk.NORMAL)
        self.btn_run.configure(image=self.run_image)
        self.btn_stop.configure(state=tk.DISABLED)
        self.output_results()
        #self.update_chart()

    def output_results(self):
        # Find different configurations:
        configurations = {}
        for result in self.all_results:
            signature = result.learner + str(result.parameters)
            if signature in configurations:
                configurations[signature].append(result.metrics.score)
            else:
                configurations[signature] = [result.metrics.score]
        #print(configurations)
        #print(len(configurations.keys()))
        #from ..evaluate.evaluate import tukey, p_stud  # , get_statistics, tukey, q_stud, p_stud
        datapoints = configurations.values()
        #print("means: {0}\nvariances: {1}\nsizes: {2}"
        #     "".format([np.mean(evaluation) for evaluation in datapoints],
        #                [np.var(evaluation) for evaluation in datapoints],
        #                [len(evaluation) for evaluation in datapoints]))
        k = len(datapoints)
        if k < 2:
            return
        df = np.max([len(evaluation) for evaluation in datapoints]) - 1
        if df < 3:
            return
        #print(get_statistics(configurations.values()))
        #qs = tukey(configurations.values())
        #print([p_stud(q, k, df) for q in qs])
        #print("q = {0}".format(q_stud(1 - self.alpha.get(), k, df)))
        #self.subplot.cla()
        #self.summary_plot = self.subplot.scatter([np.mean(configuration) for configuration in configurations.values()], np.arange(len(configurations.values())))
        #self.plot.draw()

    def output_results_(self):
        # calculate clusters:

        matrix = np.array([result.parameters.values() for result in self.all_results])
        print(matrix)
        print(self.all_results[0].parameters.keys())
        matrix = matrix[:, (1, 2, 3, 4, 6)]
        matrix = np.float_(matrix)
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=0.1).fit(matrix)
        #print(db.core_sample_indices_, db.labels_)
        a = np.argsort(db.labels_)
        print(a)
        print(db.labels_)
        labels = np.atleast_2d(db.labels_)
        print(np.hstack((matrix, labels.T)))
        colors = np.array(['#ff0000', '#00ffff', '#0000ff', '#f00f0f', '#ffff00', 'cc929f', '#000000'])
        self.summary_plot = self.subplot.scatter(matrix[:, 2], matrix[:, 4], c=colors[np.int_(db.labels_)])
        self.plot.draw()

        # TODO Get best per cluster...
        '''
        with open('/tmp/all_results.csv', 'w') as output_file:
            for result in self.all_results:
                output_file.write("{0}, {1}, {2}\n".format(result.metrics.score, hash(str(result.learner)), hash(str(result.parameters))))
        '''

    def update_chart_(self):
        """Update the chart with the ranking of the results."""
        max_values = 1000  # TODO parametrize this
        top_level_children = self.tree_rank.get_children()
        learner_results = []
        learner_topn = []
        degrees = []
        gammas = []
        coef0s = []
        for learner_branch in top_level_children[1:]:
            learner_entries = self.tree_rank.get_children(learner_branch)[:max_values]
            learner_name = self.tree_rank.item(learner_branch)['text'][:-len('Classifier')]
            learner_scores = np.array([float(self.tree_rank.item(i)['values'][-1]) for i in learner_entries])
            for i in learner_entries:
                params = self.tree_rank.item(i)['values'][-2]
                params = params.split()
                degrees.append(float(params[5][:-1]))
                gammas.append(float(params[13][:-1]))
                coef0s.append(float(params[7][:-1]))
            learner_results.append((learner_name, np.mean(learner_scores), np.std(learner_scores)))
            learner_topn.append((learner_name, learner_scores[:max_values]))
        #self.summary_plot = None
        #self.subplot.cla()
        self.summary_plot = self.subplot.scatter(coef0s[:100], gammas[:100])
        self.plot.draw()
        return
        names, means, stds = zip(*learner_results)
        names, topn = zip(*learner_topn)
        names = ["{0} ({1})".format(name, len(topn[i])) for i, name in enumerate(names)]
        try:
            num_learners = len(means)
            #if self.summary_plot is None or num_learners != len(self.summary_plot.get_children()):
            if True:  # self.summary_plot is None or num_learners != len(self.summary_plot.values()):
                self.summary_plot = None
                self.subplot.cla()
                self.subplot.axvline(0, linestyle='--', color='k', lw=1.3)
                #self.subplot.set_title('Learner performance comparison\nDataset: {0}'.format(self.dataset_name))
                self.subplot.set_title('Dataset: {0}'.format(self.dataset_name))
                #self.subplot.set_xlabel('Learner')
                self.subplot.set_xlabel('Performance index')
                self.subplot.set_xlim([-0.1, 1])
                self.subplot.yaxis.grid(False)
                self.subplot.set_autoscaley_on(True)
                #self.plot.draw()
                #self.summary_plot = self.subplot.bar(np.arange(num_learners), means,
                #                                     color=self.palette[:num_learners], width=0.6)
                #series_y = np.hstack(topn)
                #series_x = np.hstack([self.random_noise[:len(top), i] + i for (i, top) in enumerate(topn)])
                self.summary_plot = self.subplot.scatter([np.random.normal(n, 0.3, len(top)) for n in np.arange(num_learners) for top in topn], topn)
                #self.summary_plot = self.subplot.scatter(series_x, series_y)
                #self.summary_plot = self.subplot.boxplot(topn, patch_artist=True, vert=False)
                #plt.setp(self.summary_plot['boxes'], color='#5555ff', lw=2)
                #for i in range(len(self.summary_plot['whiskers']) / 2):
                #    plt.setp(self.summary_plot['whiskers'][2*i], color=self.palette[i])
                #    plt.setp(self.summary_plot['whiskers'][2*i+1], color=self.palette[i])
                self.subplot.set_yticklabels(names, size=7)
                self.summary_figure.tight_layout()

            else:
                for i in range(len(means)):
                    self.summary_plot[i].set_height(means[i])
            #self.subplot
            self.plot.draw()
        except Exception as www:
            print(www)

    def format_p_value(self, p_value):
        p_value = np.round(p_value, 4)
        if p_value <= 0.001:
            return "(<= 0.001)"
        elif p_value >= 0.9:
            return "(>= 0.9)"
        else:
            return str(p_value)

    def update_chart(self):
        # Get best configurations for each learner
        configurations = {}
        for result in self.all_results:
            signature = result.learner + '|' + str(result.parameters)
            if signature in configurations:
                configurations[signature].append(result.metrics.score)
            else:
                configurations[signature] = [result.metrics.score]

        best_configs = {}
        for signature, configuration in iteritems(configurations):
            learner, parameters = signature.split('|')
            config_mean = np.mean(configuration)
            if learner in best_configs:
                if config_mean >= np.mean(best_configs[learner]):
                    best_configs[learner] = configuration
            else:
                best_configs[learner] = configuration
        # gets very best
        best_so_far = None
        for learner, config in iteritems(best_configs):
            if best_so_far is None:
                best_so_far = learner
            else:
                current_mean = np.mean(best_configs[learner])
                print(current_mean)
                best_mean = np.mean(best_configs[best_so_far])
                if current_mean > best_mean:
                    best_so_far = learner
        from ..evaluate.evaluate import tukey, p_stud
        datapoints = best_configs.values()
        k = len(datapoints)
        df = np.max([len(evaluation) for evaluation in datapoints]) - 1
        pvalues = {}
        if k >= 2 and df > 3:
        #print(get_statistics(configurations.values()))
            qs = tukey(best_configs.values())
            pvalues = {learner: np.round(p_stud(qs[i], k, df), 4) for i, (learner, config) in enumerate(iteritems(best_configs))}
            print("p-values ", pvalues)
        #print(qs)

        """Update the chart with the ranking of the results."""
        max_values = 1000  # TODO parametrize this
        top_level_children = self.tree_rank.get_children()
        learner_results = []
        learner_topn = []
        for learner_branch in top_level_children[1:]:
            learner_entries = self.tree_rank.get_children(learner_branch)[:max_values]
            learner_name = self.tree_rank.item(learner_branch)['text'][:-len('Classifier')]
            learner_scores = np.array([float(self.tree_rank.item(i)['values'][-1]) for i in learner_entries])
            learner_results.append((learner_name, np.mean(learner_scores), np.std(learner_scores)))
            learner_topn.append((learner_name, learner_scores[:max_values]))
        names, means, stds = zip(*learner_results)
        names, topn = zip(*learner_topn)
        print("learner names", names)
        alphamin = self.alpha.get()
        accept = [pvalues.get(name + 'Classifier') >= alphamin for i, name in enumerate(names)]
        accept_ = [pvalues.get(name + 'Classifier') for i, name in enumerate(names)]
        bestindex = names.index(best_so_far[:-len('Classifier')])
        names = ["{0} ({1}) {2}".format(name, len(topn[i]), "p-value " + self.format_p_value(pvalues.get(name + 'Classifier')) if name + 'Classifier' != best_so_far else '(*)') for i, name in enumerate(names)]
        print("accept", accept)
        print("accept_", accept_)

        self.params = {'title': self.dataset_name,
                       'data': topn,
                       'means': means,
                       'accept': accept,
                       'names': names,
                       'pvalues': pvalues.values(),
                       'bestindex': bestindex}
        self.display_chart(**self.params)

    def display_chart(self, title, data, means, accept, names, pvalues, bestindex):
        # control updating only when there are changes
        accept = np.array(pvalues) >= self.alpha.get()
        try:
            #num_learners = len(means)
            #if self.summary_plot is None or num_learners != len(self.summary_plot.get_children()):
            if True:  # self.summary_plot is None or num_learners != len(self.summary_plot.values()):
                #self.summary_plot = None
                self.subplot.cla()
                self.subplot.axvline(0, linestyle='--', color='k', lw=1.2)
                self.subplot.set_title('Dataset: {0}'.format(title))
                #self.subplot.set_xlabel('Learner')
                self.subplot.set_xlabel('Performance index')
                self.subplot.set_xlim([-0.1, 1])
                self.subplot.yaxis.grid(False)
                self.subplot.set_autoscaley_on(True)
                self.summary_plot = self.subplot.boxplot(data, patch_artist=True, vert=False)
                plt.setp(self.summary_plot['whiskers'], color='#888888', linestyle='-')
                yticklabels = plt.getp(self.subplot, 'yticklabels')
                self.subplot.plot(means, np.arange(1, len(means) + 1), 'ro', color='w', marker='^', markeredgecolor='k', markersize=5)
                for i, box in enumerate(self.summary_plot['boxes']):
                    if accept[i]:
                        plt.setp(box, color=self.palette[4], edgecolor='#888888', lw=0.5)
                    else:
                        plt.setp(box, color='#b7b7b7', edgecolor='#888888', lw=0.5)
                    if yticklabels is not None:
                        if not accept[i]:
                            yticklabels[i].set_alpha(0.5)
                plt.setp(self.summary_plot['caps'], color='#333333')
                plt.setp(self.summary_plot['fliers'], color='#555555')
                plt.setp(self.summary_plot['medians'], color='#222222', linewidth=1)
                self.subplot.set_yticklabels(names, size=7)
                self.pvalue_axis.set_yticks(self.subplot.get_yticks())
                self.pvalue_axis.yaxis.grid(False)
                self.pvalue_axis.set_yticklabels(pvalues, size=5)
                self.pvalue_axis.set_ybound(self.subplot.get_ybound())
                if yticklabels is not None:
                    yticklabels[bestindex].set_fontweight('bold')
                self.summary_figure.tight_layout()
                #  Significantly different from the best (95% confidence)
            else:
                for i in range(len(means)):
                    self.summary_plot[i].set_height(means[i])
            self.plot.draw()
        except Exception as www:
            print(www)

    def update_chart_old(self):
        """Update the chart with the ranking of the results."""
        max_values = 1000  # TODO parametrize this
        top_level_children = self.tree_rank.get_children()
        learner_results = []
        learner_topn = []
        for learner_branch in top_level_children[1:]:
            learner_entries = self.tree_rank.get_children(learner_branch)[:max_values]
            learner_name = self.tree_rank.item(learner_branch)['text'][:-len('Classifier')]
            learner_scores = np.array([float(self.tree_rank.item(i)['values'][-1]) for i in learner_entries])
            learner_results.append((learner_name, np.mean(learner_scores), np.std(learner_scores)))
            learner_topn.append((learner_name, learner_scores[:max_values]))
        names, means, stds = zip(*learner_results)
        names, topn = zip(*learner_topn)
        try:
            #self.subplot.bar(np.arange(len(means)), means, color='rgbkmy')  # , yerr=stds)
            num_learners = len(means)
            #if self.summary_plot is None or num_learners != len(self.summary_plot.get_children()):
            if self.summary_plot is None or num_learners != len(self.summary_plot.values()):
                self.summary_plot = None
                self.subplot.cla()
                #self.subplot.set_title('Learner performance comparison')
                #self.subplot.set_xlabel('Learner')
                self.subplot.set_ylabel('Performance index')
                self.subplot.set_ylim([-0.1, 1])
                self.subplot.xaxis.grid(False)
                self.subplot.set_autoscalex_on(True)
                #self.plot.draw()
                #self.summary_plot = self.subplot.bar(np.arange(num_learners), means,
                #                                     color=self.palette[:num_learners], width=0.6)
                #series_y = np.hstack(topn)
                #series_x = np.hstack([self.random_noise[:len(top), i] + i for (i, top) in enumerate(topn)])
                #self.summary_plot = self.subplot.scatter([ np.random.normal(n, 0.3, len(top)) for n in np.arange(num_learners) for top in topn], topn)
                #self.summary_plot = self.subplot.scatter(series_x, series_y)
                self.summary_plot = self.subplot.boxplot(topn, patch_artist=True)
                plt.setp(self.summary_plot['whiskers'], color='#888888', linestyle='-')
                #plt.setp(self.summary_plot['boxes'], color='#5555ff', lw=2)
                for i, box in enumerate(self.summary_plot['boxes']):
                    plt.setp(box, color=self.palette[i], edgecolor='#888888', lw=0.5)
                #for i in range(len(self.summary_plot['whiskers']) / 2):
                #    plt.setp(self.summary_plot['whiskers'][2*i], color=self.palette[i])
                #    plt.setp(self.summary_plot['whiskers'][2*i+1], color=self.palette[i])
                plt.setp(self.summary_plot['caps'], color='#333333')
                plt.setp(self.summary_plot['fliers'], color='#555555')
                plt.setp(self.summary_plot['medians'], color='#222222', linewidth=1)
                #self.subplot.set_xticks(np.arange(num_learners) + 0.3)
                self.subplot.set_xticklabels(names, size=7)
                self.summary_figure.autofmt_xdate()
                #self.subplot.set_xlim([self.subplot.get_xlim()[0]-0.3, self.subplot.get_xlim()[1]-0.3])
            else:
                for i in range(len(means)):
                    self.summary_plot[i].set_height(means[i])
            #self.subplot
            self.plot.draw()
        except Exception as www:
            print(www)

    def run_command(self):
        if self.process_state == 'STOPPED':
            self.process_file()
            self.process_state = 'RUNNING'
            self.btn_stop.configure(state=tk.NORMAL)
            self.btn_open.configure(state=tk.DISABLED)
            self.btn_run.configure(image=self.pause_image)
        elif self.process_state == 'RUNNING':  # Pause
            self.command_queue.put("PAUSE")
            self.process_state = 'PAUSED'
            self.btn_run.configure(image=self.run_image)
        else:  # Resume
            self.command_queue.put("START")
            self.process_state = 'RUNNING'
            self.btn_stop.configure(state=tk.NORMAL)
            self.btn_open.configure(state=tk.DISABLED)
            self.btn_run.configure(image=self.pause_image)

    def process_file(self):
        self.reset_tree()
        Log.write("Running [SALT] in gui mode with options\n"
                  "{options}".format(options=format_dict(self.options)))

        # === Dataset ===
        settings = Settings.load_or_create_config()
        holdout = settings['Global'].as_float('holdout')
        train_set, hold_out_set = self.load_dataset(self.dataset_type.get(), holdout)
        self.dataset_name = train_set.DESCR
        print("Training with {0} points, {1} points held-out".format(len(train_set.data), len(hold_out_set.data)))
        if train_set is None:
            return  # break?

        # === Settings ===

        crossvalidation = settings['Global'].get('crossvalidation', '10')
        local_cores = settings['Global'].as_int('localcores')
        node_path_list = settings['Global'].get('nodes', ['localhost'])
        timeout = settings['Global'].as_int('timeout')
        classifier_settings = settings.get('Classifiers')
        regressor_settings = settings.get('Regressors')

        xval_rep, xval_folds = Settings.get_crossvalidation(crossvalidation)
        nodes = Settings.read_nodes(node_path_list)

        train_set.initialize(xval_rep, xval_folds)

        # === Learners ===
        default_classifiers = None
        default_regressors = None
        optimizer = 'ShrinkingHypercubeOptimizer'
        if classifier_settings:
            default_classifiers = [AVAILABLE_CLASSIFIERS[key] for key in classifier_settings
                                   if classifier_settings[key].get('enabled', False) in ('True', '$True')]
        if regressor_settings:
            default_regressors = [AVAILABLE_REGRESSORS[key] for key in regressor_settings
                                  if regressor_settings[key].get('enabled', False)]
        learners = default_regressors if train_set.is_regression else default_classifiers
        if len(learners) == 0:
            print("No learners selected")
            return

        # === Metrics ===
        metrics = []
        if metrics is None:  # or len(metrics) == 0:
            print("No metrics")
            return

        parameter_space = self.create_parameter_space(learners, settings, optimizer)

        self.finish_at = datetime.now() + timedelta(minutes=timeout)

        suggestion_task_manager = SuggestionTaskManager(train_set, learners, parameter_space, metrics,
                                                        time=timeout, report_exit_caller=self.report_results,
                                                        console_queue=self.message_queue,
                                                        command_queue=self.command_queue,
                                                        local_cores=local_cores,
                                                        node_list=nodes)
        self.summary_plot = None
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
            i = 0
            for node in self.tree_rank.get_children():
                if i != 0 and self.tree_rank.item(node)['text'] > eval_result.learner:
                    break
                i += 1
            #self.learner_ranks[eval_result.learner] = self.tree_rank.insert('', tk.END, text=eval_result.learner)
            if i == 0:
                i = 1
            self.learner_ranks[eval_result.learner] = self.tree_rank.insert('', i, text=eval_result.learner)
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
            #inserted_result = self.tree_rank.insert(learner_node, i, text=eval_result.learner, values=(eval_result.parameters, eval_result.metrics.score))
            self.tree_rank.insert(learner_node, i, text=eval_result.learner, values=(eval_result.parameters, eval_result.metrics.score))
        global_results = self.tree_rank.get_children(self.tree_global_rank)
        i = 0
        for result in global_results:
            score = float(self.tree_rank.item(result).values()[2][1])
            if score < eval_result.metrics.score:
                break
            i += 1
        self.tree_rank.insert(self.tree_global_rank, i, text=eval_result.learner, values=(eval_result.parameters, eval_result.metrics.score))
        self.all_results.append(eval_result)

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
                        if self.process_state == 'STOPPED':
                            self.update_chart()
                        #self.tree_rank.insert(self.tree_global_rank, tk.END, text=message.learner, values=(message.parameters, message.metrics.score))
                        #if message.learner not in self.learner_ranks:
                        #    self.learner_ranks[message.learner] = self.tree_rank.insert('', tk.END, text=message.learner)
                        #self.tree_rank.insert(self.learner_ranks[message.learner], tk.END, text=message.learner, values=(message.parameters, message.metrics.score))
                    elif type(message) is int:
                        if message == 1:
                            self.stop_processing()
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

    def update_clock(self):
        try:
            if self.process_state == 'RUNNING':
                timediff = self.finish_at - datetime.now()
                if timediff.total_seconds() >= 0:
                    timetext = "{0:02d}:{1:02d}:{2:02d}".format(timediff.seconds / 3600, timediff.seconds / 60, timediff.seconds % 60)
                    self.statusbartext.config(text=timetext)
                self.update_chart()
        except:
            pass
        self.after(1000, self.update_clock)

    def show(self):
        """Display the main window."""
        Log.write("Loading GUI with options \n{options}".format(options=format_dict(self.options)))

        #self.root.bind('<Configure>', self.on_form_event)
        self.root.mainloop()
