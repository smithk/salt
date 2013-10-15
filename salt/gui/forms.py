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
from ..learn import DEFAULT_LEARNERS
from ..suggest import SuggestionTaskManager
import itertools
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
        self.root.attributes('-topmost', True)
        self.options = options
        self.file_to_open = None
        #super(SaltMain, self).__init__(master)
        ttk.Frame.__init__(self, master)
        self.pack()
        self.setup_gui()

    def setup_gui(self):
        """Create all user controls."""
        images_path = join(dirname(__file__), 'images')
        open_image = tk.PhotoImage(file="{path}/open.gif".format(path=images_path))
        self.btn_action = ttk.Button(self, text="Do SALT", image=open_image)
        self.btn_action["command"] = self.do_something
        self.btn_action.image = open_image
        self.btn_action.pack({"side": "left"})

        self.btn_quit = ttk.Button(self)
        self.btn_quit["text"] = "eval n run"
        self.btn_quit["command"] = self.process_file  # self.quit
        self.btn_quit.pack({"side": "left"})
        ttk.Style().theme_use('clam')

    def do_something(self):  # pylint: disable=R0201
        """Do something to test button functionality."""
        self.file_to_open = filedialog.askopenfilename(filetypes={("Arff files", "*.arff")})
        if self.file_to_open:
            print("open file {thefile}".format(thefile=self.file_to_open))

    def load_dataset(self):
        if self.file_to_open:
            arff_reader = ArffReader(self.file_to_open)
            dataset = arff_reader.read_file()

            return dataset

    def create_default_parameters(self, learners):
        parameter_dict = {learner.__name__: learner.create_default_params() for learner in learners}
        return parameter_dict

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


    def process_file(self):
        Log.write("Running [SALT] in gui mode with options\n{options}".format(options=format_dict(self.options)))

        # === Dataset ===
        dataset = self.load_dataset()
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

        parameters = self.create_default_parameters(learners)

        suggestion_task_manager = SuggestionTaskManager(dataset, learners, parameters, metrics,
                                                        time=10, report_exit_caller=self.report_results)
        suggestion_task_manager.run_tasks()

    def show(self):
        """Display the main window."""
        Log.write("Loading GUI with options \n{options}".format(
            options=format_dict(self.options)))

        self.root.mainloop()
        #self.root.destroy()
