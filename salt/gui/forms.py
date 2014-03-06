"""
The :mod:`salt.gui.forms` module provides an implementation of the forms needed
for the user to interact with salt on a graphical interface.
"""

import os
from os.path import join, dirname
from six import PY3, iteritems
from six.moves import tkinter as tk, tkinter_tkfiledialog as FileDialog
from multiprocessing import Queue, Lock
from Queue import Empty as EmptyQueue
from collections import OrderedDict
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
if PY3:
    from tkinter import ttk
else:
    import ttk
from ..utils.debug import Log
from ..utils.strings import format_dict
from ..IO.readers import ArffReader
from ..learn import AVAILABLE_CLASSIFIERS, AVAILABLE_REGRESSORS, create_parameter_space
from ..suggest import SuggestionTaskManager
from ..evaluate import EvaluationResults
from .controls import ToolTip
from ..evaluate.evaluate import p_stud, get_statistics, format_p_value


class SaltMain(ttk.Frame):
    """Main SALT window."""
    def __init__(self, options, master=None):
        # Initialize global variables
        self.options = options
        self.all_results = []
        self.default_results = {}
        self.lock = Lock()
        self.dataset_path = None
        self.dataset_name = ''
        self.xval_repetitions = 1
        self.xval_folds = 1
        self.message_queue = Queue()  # Receives messages from tasks
        self.command_queue = Queue()  # Sends commands to tasks
        self.process_state = 'STOPPED'
        self.update_plot_required = False
        self.max_results = 10

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Suggest-a-Learner Toolbox")
        self.root.attributes('-topmost', True)
        window_width= self.root.winfo_screenwidth()
        window_height=self.root.winfo_screenheight()
        window_width = 800
        window_height = 600
        self.root.geometry("{0}x{1}".format(int(window_width), int(window_height)))
        ttk.Frame.__init__(self, master)
        self.pack(fill=tk.BOTH, expand=1)
        self.alpha = tk.DoubleVar()
        self.show_default = tk.BooleanVar()
        self.show_n = tk.BooleanVar()
        self.setup_gui()
        self.alpha.trace("w", self.call_update_chart)
        self.show_default.trace("w", self.call_update_chart)
        self.show_n.trace("w", self.call_update_chart)

    def call_update_chart(self, *args):
        if self.dataset_path is not None:
            self.update_chart()

    # ===== GUI SETUP =====

    def setup_toolbar(self):
        ''' Create and organize toolbar controls.'''
        images_path = join(dirname(__file__), 'images')

        open_image = tk.PhotoImage(file=join(images_path, "open.gif"))
        run_image = tk.PhotoImage(file=join(images_path, "gears_run.gif"))
        pause_image = tk.PhotoImage(file=join(images_path, "gears_pause.gif"))
        stop_image = tk.PhotoImage(file=join(images_path, "gears_stop.gif"))

        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        btn_open = ttk.Button(toolbar, text="Open dataset", image=open_image)
        btn_open["command"] = self.show_open_file
        btn_open.image = open_image
        btn_open.pack(side=tk.LEFT)
        ToolTip(btn_open, msg="Select your dataset")

        btn_run = ttk.Button(toolbar, image=run_image, state=tk.DISABLED)
        btn_run.text = "Run"
        btn_run["command"] = self.run_command  # self.quit
        btn_run.image = run_image
        btn_run.pack(side=tk.LEFT, padx=5)
        ToolTip(btn_run, msg="Start optimization")

        btn_stop = ttk.Button(toolbar, image=stop_image, state=tk.DISABLED)
        btn_stop.text = "Stop"
        btn_stop["command"] = self.stop_processing
        btn_stop.image = stop_image
        btn_stop.pack(side=tk.LEFT)
        ToolTip(btn_stop, msg="Stop optimization")

        self.dataset_type = tk.BooleanVar()
        self.dataset_type.set(False)
        dataset_type_container = ttk.Frame(toolbar)
        dataset_type_container.pack(side=tk.RIGHT)
        btns_dataset_type_classif = \
            ttk.Radiobutton(dataset_type_container, text='Classification',
                            variable=self.dataset_type, value=False)
        btns_dataset_type_regr = \
            ttk.Radiobutton(dataset_type_container, text='Regression',
                            variable=self.dataset_type, value=True)
        btns_dataset_type_classif.pack(side=tk.TOP)
        btns_dataset_type_regr.pack(side=tk.LEFT)

        # Store references for controls to be accessed elsewhere
        self.run_image = run_image
        self.pause_image = pause_image
        self.btn_open = btn_open
        self.btn_run = btn_run
        self.btn_stop = btn_stop

    def setup_statusbar(self):
        ''' Create and organize statusbar controls.'''
        self.status_msg = tk.StringVar()
        self.status_msg.set("Ready!")
        statusbar = ttk.Frame(self)
        statusbartext = ttk.Label(statusbar, textvariable=self.status_msg)
        statusbartext.pack(side=tk.LEFT, fill=tk.X)
        sizegrip = ttk.Sizegrip(statusbar)

        self.selected_optimizer = tk.StringVar()
        optimizer_values = ('None (run with default parameters)', 'Random search', 'Shrinking hypercube')
        optimizer_list = ttk.Combobox(statusbar, values=optimizer_values,
                                      textvariable=self.selected_optimizer,
                                      width=30, state='readonly')
        optimizer_list.current(0)  # Default optimizer

        optimizer_label = ttk.Label(statusbar, text='Optimization method: ')
        sizegrip.pack(side=tk.RIGHT)
        optimizer_list.pack(side=tk.RIGHT)
        optimizer_label.pack(side=tk.RIGHT)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)

    def setup_console_frame(self):
        console_frame = ttk.Frame(self)
        console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1, padx=2, pady=2)

        console = ttk.Tkinter.Text(console_frame, bg="black", fg="#aaa", font=("Ubuntu Mono", 9))
        console_scroll = ttk.Scrollbar(console_frame)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        console_scroll.config(command=console.yview)
        console.config(yscrollcommand=console_scroll.set)
        console.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Store references for controls to be accessed elsewhere
        self.console_frame = console_frame
        self.console = console
        return self.console_frame

    def setup_ranking_frame(self):
        # Learner results tree view
        ranking_frame = ttk.Frame(self)
        ranking_frame.pack(fill=tk.BOTH, expand=1, padx=2, pady=2)

        tree_rank = ttk.Treeview(ranking_frame, columns=('configuration', 'score'))
        tree_rank.heading("#0", text="Learner")
        tree_rank.heading(0, text="Configuration")
        tree_rank.heading(1, text="Score")
        self.tree_global_rank = tree_rank.insert('', 1, text="Global ranking")
        self.learner_ranks = {}
        ranking_scroll = ttk.Scrollbar(ranking_frame)
        ranking_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        ranking_scroll.config(command=tree_rank.yview)
        tree_rank.config(yscrollcommand=ranking_scroll.set)
        tree_rank.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Store references for controls to be accessed elsewhere
        self.ranking_frame = ranking_frame
        self.tree_rank = tree_rank
        return self.ranking_frame

    def setup_visualization_frame(self):
        plot_frame = ttk.Frame(self)
        summary_figure = Figure(figsize=(5, 4), dpi=100)
        plot_canvas = FigureCanvasTkAgg(summary_figure, master=plot_frame)
        # summary_figure.suptitle('[figure title here] $\\alpha$')

        subplot = summary_figure.add_subplot(111)
        subplot.format_coord = lambda x, y: ''
        subplot.set_ylim([0, 1])
        subplot.xaxis.grid(False)
        pvalue_axis = subplot.twinx()
        pvalue_axis.format_coord = lambda x, y: ''
        pvalue_axis.set_ylim([0, 1])
        pvalue_axis.xaxis.grid(False)
        self.summary_plot = None
        subplot.set_ylabel('Score')

        plot_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        plot_canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)
        plot_toolbar = CustomNavToolbar(plot_canvas, plot_frame, alpha_var=self.alpha,
                                        show_default_var=self.show_default, show_n_var=self.show_n)
        plot_toolbar.update()
        plot_frame.pack()
        summary_figure.tight_layout()

        self.plot_frame = plot_frame
        self.summary_figure = summary_figure
        self.subplot = subplot
        self.pvalue_axis = pvalue_axis
        return self.plot_frame

    def setup_settings_frame(self):
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
        return self.settings_frame

    def setup_main_panel(self):
        '''Create and organize all controls that are shown on the main panel.'''
        self.notebook = ttk.Notebook(self)

        console_frame = self.setup_console_frame()
        ranking_frame = self.setup_ranking_frame()
        visualization_frame = self.setup_visualization_frame()
        settings_frame = self.setup_settings_frame()

        self.notebook.add(ranking_frame, text="Ranking")
        self.notebook.add(visualization_frame, text="Visualization")
        self.notebook.add(console_frame, text="Console")
        self.notebook.add(settings_frame, text="Settings")
        self.notebook.pack(fill=tk.BOTH, expand=1)

    def setup_gui(self):
        """Create all user controls."""
        self.setup_toolbar()
        self.setup_main_panel()
        self.setup_statusbar()

        ttk.Style().theme_use('clam')

    def show(self):
        """Display the main window."""
        Log.write("Loading GUI with options \n{options}".format(options=format_dict(self.options)))

        self.root.bind('<Configure>', self.update_plot_dimensions)
        self.root.mainloop()

    def update_plot_dimensions(self, event):
        # Fits plot scale after window resizing events
        self.summary_figure.tight_layout()

    # ===== EVENTS =====

    def show_open_file(self):
        '''Show the 'Open file' dialog.'''
        # TODO Load filetypes from available file readers.
        avail_filetypes = (("Arff files", "*.arff"),)
        self.dataset_path = FileDialog.askopenfilename(filetypes=avail_filetypes)
        if self.dataset_path is not None:
            self.btn_run.configure(state=tk.NORMAL)
            self.message_queue.put("open file {thefile}\n".format(thefile=self.dataset_path))

    def stop_processing(self):
        self.command_queue.put("STOP")
        self.process_state = 'STOPPED'
        self.btn_open.configure(state=tk.NORMAL)
        self.btn_run.configure(image=self.run_image)
        self.btn_stop.configure(state=tk.DISABLED)

    def get_data_to_analyze(self):
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
        return best_configs

    def update_chart(self):
        data = self.get_data_to_analyze()
        #if len(data) <= 1:  # TODO Handle the case one learner
        #    return
        means = np.array([np.mean(model) for model in data.values()])
        ordering = np.argsort(data.keys())
        # Store data sorted by name
        means = means[ordering]
        data = OrderedDict(zip(np.array(data.keys())[ordering], np.array(data.values())[ordering]))
        k = len(means)
        sizes = [len(model) for model in data.values()]
        df = np.max(sizes) - 1
        pvalues = []
        if k >= 2 and df > 3:
            #qs = tukey(data.values())
            qs = get_statistics(data.values())
            pvalues = [np.round(p_stud(q, k, df), 4) for q in qs]
        #print(data, best_index, data.values()[best_index], np.mean(data.values()[best_index]), means)
        stats = {'means': means,
                 'pvalues': pvalues,
                 'data': data.values(),
                 'names': data.keys(),
                 'sizes': sizes
                 }
        title = self.dataset_name
        self.display_data(title, stats)
        self.update_plot_required = False

    def display_chart(self, title, data, means, accept, names, pvalues, sizes, bestindex):
        # control updating only when there are changes
        accept = np.array(pvalues) >= self.alpha.get()
        plot_data = '?'
        show_defaults = self.show_default.get()
        try:
            #num_learners = len(means)
            #if self.summary_plot is None or num_learners != len(self.summary_plot.get_children()):
            if True:  # self.summary_plot is None or num_learners != len(self.summary_plot.values()):
                #self.summary_plot = None
                n = self.xval_repetitions * self.xval_folds
                self.setup_subplot(title, n)
                positions = np.arange(1, len(data) + 1)
                #positions = np.vstack((positions, positions + 0.5))
                default_data = self.default_results.values()  # [np.array(row) - 0.2 for row in data]
                default_means = [np.mean(row) for row in default_data]
                #self.summary_plot = self.subplot.boxplot(plot_data, positions=positions, widths=widths,  patch_artist=True, vert=False)
                if show_defaults:
                    ghostcolor = '#aaaaaa'
                    edgecolor = '#efefef'
                    mediancolor = '#efefef'
                    alpha = 0.4
                    a = self.subplot.boxplot(default_data, positions=positions + 0.325, widths=0.25, patch_artist=True, vert=False, notch=True)
                    plt.setp(a['whiskers'], color=ghostcolor, linestyle='-', alpha=alpha, linewidth=1.2)
                    plt.setp(a['boxes'], color=ghostcolor, edgecolor=edgecolor, alpha=alpha)
                    plt.setp(a['caps'], color=ghostcolor, alpha=alpha, linewidth=1.2)
                    plt.setp(a['fliers'], color=ghostcolor, alpha=alpha, linewidth=1.2, marker='o', markersize=3.2, markeredgecolor=ghostcolor)
                    plt.setp(a['medians'], color=mediancolor, linewidth=1.2, alpha=alpha)
                    boxplot_width = 0.5
                #else:
                boxplot_width = 0.9
                self.subplot.axvline(max(means), linestyle=':', color='w', lw=1.2, zorder=0)  # Best mean
                offset = 0
                if show_defaults:
                    offset = 0.15
                    positions = positions - offset
                self.summary_plot = self.subplot.boxplot(data, positions=positions, widths=boxplot_width - offset * 2, patch_artist=True, vert=False, notch=True)
                yticks = self.subplot.get_yticks()
                yticks = np.hstack((yticks, yticks[-1] + 1))[::2]
                self.subplot.barh(yticks - 0.5 + offset, [1.1] * len(yticks), height=1, left=-0.1, color='#cdcdcd', linewidth=0, alpha=0.25, zorder=0)
                self.setup_legend()
                self.subplot.set_ylim([0.4, len(data) + 0.6])
                self.setup_boxplot(names, sizes, pvalues, accept, bestindex)
                self.plot_means(means, vertical_offset=-offset)
                if show_defaults:
                    self.plot_means(default_means, vertical_offset=0.325, alpha=0.4, boxheight=0.05)
                self.summary_figure.tight_layout(rect=(0, 0.07, 1, 1))
                #  Significantly different from the best (95% confidence)
            else:
                for i in range(len(means)):
                    self.summary_plot[i].set_height(means[i])
            self.summary_figure.canvas.draw()
        except Exception as exc:
            print("Problem with plot: {0} on {1}".format(exc, plot_data))

    def display_data(self, title, stats, alpha=None):
        if alpha is None:
            alpha = self.alpha.get()
        bestindex = np.argmax(stats['means'])
        accept = np.array(stats['pvalues']) >= alpha
        self.display_chart(title, stats['data'], stats['means'], accept, stats['names'], stats['pvalues'], stats['sizes'],  bestindex)

    def setup_subplot(self, title, n):
        self.subplot.cla()
        self.subplot.axvline(0, linestyle='--', color='k', lw=1.2)
        #self.subplot.set_title('Dataset: {0}. $\\alpha$ threshold: {1}'.format(title, self.alpha.get()))
        self.subplot.set_title(u'{0}. $\\alpha={1}$, $n={2}$'.format(title, self.alpha.get(), n))
        self.subplot.set_xlabel('Performance index')
        self.subplot.set_xlim([-0.1, 1])
        self.subplot.yaxis.grid(False)
        #self.subplot.set_autoscaley_on(True)

    # ===== PLOT HANDLING =====

    def plot_means(self, means, vertical_offset=0, alpha=1, boxheight=0.08):
        '''Plot the means.'''
        height = self.subplot.transData.transform([(0, boxheight), (0, 0)])
        height = height[0, 1] - height[1, 1]
        #self.subplot.plot(means, np.arange(1, len(means) + 1) + vertical_offset, 'ro', color='w', marker="$|$", markeredgecolor='k', markersize=8)
        self.subplot.plot(means, np.arange(1, len(means) + 1) + vertical_offset, 'ro', color='w', marker=[(0, -.5), (0, .5)], markeredgecolor='k', linewidth=2, markersize=height, markeredgewidth=2, alpha=alpha)
        #self.subplot.set_title('Dataset: {0}. $\\alpha$ threshold: {1}'.format(title, self.alpha.get()))

    def setup_legend(self):
        '''Create the legend for the summary plot.'''
        artists = [plt.Line2D((0, 0), (2, 2), linewidth=0,marker=[(0, -.5), (0, .5)], markeredgecolor='k', markeredgewidth=2),
                   #           markersize=5, markerfacecolor='w'),
                   plt.Rectangle((0, 0), 1, 1, facecolor='#aaaaaa', edgecolor='#efefef', alpha=0.4),
                   plt.Rectangle((0, 0), 1, 1, facecolor='#b7b7b7'),
                   plt.Rectangle((0, 0), 1, 1, facecolor='#3387bc')]
        legend = self.subplot.legend(artists,
                                     ['mean', 'default parameters', 'significant difference',
                                      'non-significant difference'],
                                     ncol=4, bbox_to_anchor=(1, -0.14), prop={'size': 8},
                                     numpoints=1, borderaxespad=0)
        legend_frame = legend.get_frame()
        legend_frame.set_linewidth(0)
        legend_frame.set_alpha(0.75)

    def setup_boxplot(self, names, sizes, pvalues, accept, bestindex):
        lw = 1.4
        plt.setp(self.summary_plot['whiskers'], color='#333333', linestyle='-', lw=lw)
        yticklabels = plt.getp(self.subplot, 'yticklabels')
        if len(accept) > 0:
            for i, box in enumerate(self.summary_plot['boxes']):
                if i >= len(accept):
                    plt.setp(box, facecolor="#ffffff", edgecolor='#888888', lw=0.5, alpha=0.7)
                    continue
                if accept[i]:
                    plt.setp(box, facecolor="#3387bc", edgecolor='#3387bc', lw=0)
                else:
                    plt.setp(box, facecolor='#b0b0b0', edgecolor='#b7b7b7', lw=0)
                if yticklabels is not None:
                    if not accept[i]:
                        yticklabels[i].set_alpha(0.5)
        plt.setp(self.summary_plot['caps'], color='#333333', lw=lw)
        plt.setp(self.summary_plot['fliers'], color='#333333', markeredgecolor='#333333', marker='o', markersize=3.2)
        plt.setp(self.summary_plot['medians'], color='#e5e5e5', linewidth=1.7)
        if self.show_n.get():
            names = ["{0} ({1})".format(name[:-len("Classifier")], sizes[i]) for i, name in enumerate(names)]
        else:
            names = [name[:-len("Classifier")] for name in names]

        for cap in self.summary_plot['caps']:
            shrink = 0.5
            bounds = cap.get_ydata()
            width = bounds[1] - bounds[0]
            newbounds = bounds[0] + shrink * width / 2., bounds[1] - shrink * width / 2.
            cap.set_ydata(newbounds)

        self.subplot.set_yticklabels(names, size=7)
        self.pvalue_axis.set_yticks(self.subplot.get_yticks())
        self.pvalue_axis.yaxis.grid(False)
        display_p_values = [format_p_value(pvalue) if i != bestindex else '' for (i, pvalue) in enumerate(pvalues)]
        self.pvalue_axis.set_yticklabels(display_p_values, size=6)
        #self.subplot.set_ylim([.5, len(data) + 1])
        self.pvalue_axis.set_ybound(self.subplot.get_ybound())
        if yticklabels is not None:
            yticklabels[bestindex].set_fontweight('bold')

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
        self.all_results = []
        self.default_results = {}
        Log.write("Running [SALT] in gui mode with options\n"
                  "{options}".format(options=format_dict(self.options)))

        # === Dataset ===
        settings = Settings.load_or_create_config()
        holdout = settings['Global'].as_float('holdout')
        train_set, hold_out_set = ArffReader.load_dataset(self.dataset_path, self.dataset_type.get(), holdout)
        self.dataset_name = train_set.DESCR
        print("Training with {0} points, {1} points held-out".format(len(train_set.data), len(hold_out_set.data)))
        if train_set is None:
            return  # break?

        # === Settings ===

        crossvalidation = settings['Global'].get('crossvalidation', '10')
        local_cores = settings['Global'].as_int('localcores')
        node_path_list = settings['Global'].get('nodes', ['localhost'])
        timeout = settings['Global'].as_int('timeout')
        max_jobs = settings['Global'].as_int('maxprocesses')
        classifier_settings = settings.get('Classifiers')
        regressor_settings = settings.get('Regressors')

        self.max_results = settings['GUI'].as_int('maxresults')

        self.xval_repetitions, self.xval_folds = Settings.get_crossvalidation(crossvalidation)
        nodes = Settings.read_nodes(node_path_list)

        train_set.initialize(self.xval_repetitions, self.xval_folds)

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

        parameter_space = create_parameter_space(learners, settings)

        self.finish_at = datetime.now() + timedelta(minutes=timeout)
        optimizer = self.selected_optimizer.get()

        suggestion_task_manager = SuggestionTaskManager(train_set, learners, parameter_space,
                                                        metrics, time=timeout,
                                                        report_exit_caller=self.report_results,
                                                        console_queue=self.message_queue,
                                                        command_queue=self.command_queue,
                                                        local_cores=local_cores,
                                                        node_list=nodes, optimizer=optimizer,
                                                        max_jobs=max_jobs)
        self.summary_plot = None
        suggestion_task_manager.run_tasks(wait=False)
        # Start processes
        self.update_content()
        self.update_clock()

    def reset_tree(self):
        for learner in self.learner_ranks.values():
            self.tree_rank.delete(learner)
        self.tree_rank.delete(self.tree_global_rank)
        self.tree_global_rank = self.tree_rank.insert('', 1, text="Global ranking")
        self.learner_ranks = {}

    def add_result(self, eval_result):
        self.lock.acquire()
        top_n = self.max_results
        # Create top-level branch for classifier if it does not exist
        if eval_result.learner not in self.learner_ranks:
            i = 0
            for node in self.tree_rank.get_children():
                # Find index to insert alphabetically ordered
                if i != 0 and self.tree_rank.item(node)['text'] > eval_result.learner:
                    break
                i += 1
            if i == 0:
                i = 1
            self.learner_ranks[eval_result.learner] = self.tree_rank.insert('', i, text=eval_result.learner)
        learner_node = self.learner_ranks[eval_result.learner]
        learner_results = self.tree_rank.get_children(learner_node)
        i = 0
        for result in learner_results:
            score = float(self.tree_rank.item(result).values()[2][1])
            if score <= eval_result.metrics.score:
                break
            i += 1
        if i < top_n:
            self.tree_rank.insert(learner_node, i, text=eval_result.learner, values=(eval_result.parameters, eval_result.metrics.score))
            if len(learner_results) == top_n:
                self.tree_rank.delete(learner_results[top_n - 1])
        global_results = self.tree_rank.get_children(self.tree_global_rank)
        i = 0
        for result in global_results:
            score = float(self.tree_rank.item(result).values()[2][1])
            if score <= eval_result.metrics.score:
                break
            i += 1
        if i < top_n:
            self.tree_rank.insert(self.tree_global_rank, i, text=eval_result.learner, values=(eval_result.parameters, eval_result.metrics.score))
            if len(global_results) == top_n:
                self.tree_rank.delete(global_results[top_n - 1])
        if eval_result.parameters == {}:
            if eval_result.learner in self.default_results:
                self.default_results[eval_result.learner].append(eval_result.metrics.score)
            else:
                self.default_results[eval_result.learner] = [eval_result.metrics.score]
                ordering = np.argsort(self.default_results.keys())
                keys = np.array(self.default_results.keys())[ordering].tolist()
                values = np.array(self.default_results.values())[ordering].tolist()
                self.default_results = OrderedDict(zip(keys, values))
        #else:
        self.all_results.append(eval_result)
        self.lock.release()
        self.update_plot_required = True

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
        self.after(10, self.update_content)

    def update_clock(self):
        try:
            if self.process_state == 'RUNNING':
                timediff = self.finish_at - datetime.now()
                if timediff.total_seconds() >= 0:
                    timetext = "{0:02d}:{1:02d}:{2:02d}".format(timediff.seconds / 3600, timediff.seconds / 60 - timediff.seconds / 3600 * 60, timediff.seconds % 60)
                    self.status_msg.set(timetext)
                if self.update_plot_required:
                    self.update_chart()
        except:
            pass
        self.after(10, self.update_clock)

    def report_results(self, ranking):
        '''Print to the stdout the results of the optimization.'''
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


class CustomNavToolbar(NavigationToolbar2TkAgg):
    '''Navigation toolbar that includes alpha slider.'''
    def __init__(self, plot_canvas, plot_window, alpha_var, show_default_var, show_n_var):
        NavigationToolbar2TkAgg.__init__(self, plot_canvas, plot_window)
        self.alpha = alpha_var
        self.show_default = show_default_var
        self.show_n = show_n_var
        self.alpha_slider = self._slider(0.05, 0.9)
        self.check_container = ttk.Frame(self)
        self.check_container.pack(side=tk.LEFT)
        self.show_default_check = self._check_default()
        self.show_num_samples = self._check_show_n()
        self._message_label.pack(side=tk.LEFT)

    def _slider(self, lower, upper, default=0.05):
        sliderlabel = tk.Label(master=self, text=u'\u03b1: ')
        slider = tk.Scale(master=self, from_=lower, to=upper, orient=tk.HORIZONTAL,
                          resolution=0.01, bigincrement=0.1, bd=1, width=10, showvalue=default, variable=self.alpha)
        slider.pack(side=tk.RIGHT)
        sliderlabel.pack(side=tk.RIGHT)
        return slider

    def _check_show_n(self):
        check = tk.Checkbutton(master=self.check_container, text="Show number of samples per box", variable=self.show_n)
        check.pack(side=tk.LEFT)
        return check

    def _check_default(self):
        check = tk.Checkbutton(master=self.check_container, text="Show default configuration performances", variable=self.show_default)
        check.pack(side=tk.BOTTOM)
        return check
