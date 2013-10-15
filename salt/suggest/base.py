"""The :mod:`salt.suggest.base` module provides classes to describe suggestion tasks."""

from multiprocessing import Process, Queue, Lock, Manager
from ..learn.cross_validation import CrossValidationGroup
from ..optimize.base import SequentialOptimizer
#from ..optimize.base import RandomOptimizer
#from ..jobs import job_manager
from ..jobs.base import JobManager
from ..evaluate.metrics import Metrics
from ..evaluate.base import EvaluationResults


# TODO: provide parameter to configure folds via cmdline/settings file
FOLDS = 20


class SuggestionTask(Process):
    def __init__(self, suggestion_task_manager, learner, parameters,
                 task_queue, result_queue, lock):
        self.manager = suggestion_task_manager
        self.learner = learner
        self.parameters = parameters
        self.optimizer = SequentialOptimizer(self.parameters)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.lock = lock
        self.status = ''
        super(SuggestionTask, self).__init__(target=self.run)

    def join(self):
        print("[Task] [ {0} Exiting... ]".format(self.learner.__name__))
        #try:
        super(SuggestionTask, self).join()
        #except:
        #    pass

    def run(self):
        configuration = self.optimizer.get_next_configuration()
        while configuration is not None:
            # TODO: Read cross-validation folds from settings.
            cross_val_group = CrossValidationGroup(self.learner, configuration,
                                                   self.manager.dataset, folds=FOLDS)
            self.lock.acquire()
            try:
                self.task_queue.put(cross_val_group)
            except Exception as e:
                print(e)
            self.lock.release()
            configuration = self.optimizer.get_next_configuration()
            if not self.result_queue.empty():
                job_results = self.result_queue.get()
                if job_results:
                    self.notify_results(job_results)
            # Report completion to job manager
        self.lock.acquire()
        self.task_queue.put((self.learner.__name__, 'Finished'))
        self.lock.release()
        #print("All tasks sent. {0} waiting for poison pill".format(self.learner.__name__))
        job_results = self.result_queue.get()
        while job_results:
            self.notify_results(job_results)
            job_results = self.result_queue.get()
        self.status = 'Finished'
        #print("{0} poison pill received".format(self.learner.__name__))
        print("[Task {0} finished. Should exit now ]".format(self.learner.__name__))

    def notify_results(self, job):
        metrics = Metrics(self.manager.dataset.target, job.labels)
        evaluation_results = EvaluationResults(self.learner.__name__,
                                               job.parameters, metrics)
        self.optimizer.report_results(evaluation_results)
        self.lock.acquire()
        self.manager.add_task_results(self, True)
        self.lock.release()


class SuggestionTaskManager():
    def __init__(self, dataset, learners, parameters, metrics, time, report_exit_caller):
        self.dataset = dataset
        self.metrics = metrics
        self.time = time
        self.lock = Lock()
        self.report_exit_caller = report_exit_caller
        #self.poison_pill_sent = False
        # necessary to share memory
        self.proc_manager = Manager()
        self.statuses = self.proc_manager.dict({learner.__name__: False for learner in learners})
        self.ranking = self.proc_manager.dict({learner.__name__: None for learner in learners})
        self.finished = self.proc_manager.dict({learner.__name__: False for learner in learners})
        self.task_queue = Queue()
        self.queues = {learner.__name__: Queue() for learner in learners}
        self.suggestion_tasks = {learner.__name__:
                                 SuggestionTask(self, learner, parameters[learner.__name__],
                                                self.task_queue, self.queues[learner.__name__], self.lock)
                                 for learner in learners}
        #self.cluster = pp.Server(0, ppservers=('10.2.172.4', '10.2.164.194',))
        self.job_manager = JobManager(self.task_queue, self.queues, self.lock, self.finished)

    def all_tasks_finished(self):
        #return all(self.statuses.values())
        return all([task.status == 'Finished' for task in self.suggestion_tasks.values()])

    def run_tasks(self):
        self.job_manager.daemon = True
        self.job_manager.start()
        self.poison_pill_sent = False
        for suggestion_task in self.suggestion_tasks.values():
            suggestion_task.daemon = True
            suggestion_task.start()

        for suggestion_task in self.suggestion_tasks.values():
            if suggestion_task.is_alive():
                suggestion_task.join()
            #    print("*****************************************Alive proces")
            print("[Task Manager] [Waiting for task to finish {0}]".format(suggestion_task.learner.__name__))
            #suggestion_task.join()
            suggestion_task.result_queue.close()
            suggestion_task.result_queue.join_thread()
            print("[Task Manager] [Task {0} finished]".format(suggestion_task.learner.__name__))

        # job_manager only joins after all tasks have finished
        self.job_manager.join()
        self.report_exit_caller(self.ranking)

    def add_task_results(self, task, status):
        #self.lock.acquire()
        suggestion_task_name = task.learner.__name__
        self.statuses[suggestion_task_name] = status
        task.optimizer.evaluation_results.reverse()
        self.ranking[suggestion_task_name] = task.optimizer.evaluation_results
        if self.all_tasks_finished():
            print("[Task Manager] [ Sending poison pill to job manager ]")
            self.task_queue.put(None)
            self.poison_pill_sent = True
        #self.lock.release()
