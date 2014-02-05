"""The :mod:`salt.suggest.base` module provides classes to describe suggestion tasks."""

from Queue import Empty
from multiprocessing import Process, Queue, Lock, Manager
from ..learn.classifiers import BaselineClassifier
from ..learn.regressors import BaselineRegressor
from ..learn.cross_validation import CrossValidationGroup
from ..optimize import KDEOptimizer, ShrinkingHypercubeOptimizer, DefaultConfigOptimizer
#from ..optimize.base import RandomOptimizer
#from ..jobs import job_manager
from ..jobs import JobManager
from ..evaluate.metrics import ClassificationMetrics, RegressionMetrics
from ..evaluate import EvaluationResults
from ..utils.strings import now
from datetime import datetime, timedelta
import sys


class CommandDispatcher(Process):
    def __init__(self, command_queue, process_queues):
        self.command_queue = command_queue
        self.process_queues = process_queues
        super(CommandDispatcher, self).__init__(target=self.run)

    def run(self):
        command = None
        while command != 'STOP':
            try:
                command = self.command_queue.get(timeout=0.5)
            except Empty:
                pass
            if command is not None:
                for process_queue in self.process_queues.values():
                    process_queue.put(command)
            command = None


class SuggestionTask(Process):
    def __init__(self, suggestion_task_manager, learner, parameters,
                 task_queue, result_queue, lock, console_queue, finish_at, max_tasks, command_queue):
        self.manager = suggestion_task_manager
        self.learner = learner
        self.parameters = parameters
        #self.optimizer = SequentialOptimizer(self.parameters)
        #self.optimizer = KDEOptimizer(self.parameters)
        self.optimizer = ShrinkingHypercubeOptimizer(self.parameters)
        #self.optimizer = DefaultConfigOptimizer(self.parameters)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.lock = lock
        self.console_queue = console_queue
        self.command_queue = command_queue
        self.status = ''
        self.finish_at = finish_at
        self.max_tasks = max_tasks
        self._c = 0
        super(SuggestionTask, self).__init__(target=self.run)

    def join(self):
        super(SuggestionTask, self).join()
        print("{0} [Task] [ {1} finished ]".format(now(), self.learner.__name__))

    def _is_timeout(self):
        self._c += 1
        #print(datetime.now(), self.finish_at)
        return datetime.now() > self.finish_at
        #return self._c > 10

    timeout = property(_is_timeout)

    def run(self):
        configuration = self.optimizer.get_next_configuration()
        tasks_running = 0
        try:
            while configuration is not None:  # and not self.timeout:
                #self.finish_at = datetime.now()  # One run only (for testing)
                # TODO: Read cross-validation folds from settings.
                cross_val_group = CrossValidationGroup(self.learner, configuration,
                                                       self.manager.dataset)
                self.lock.acquire()
                try:
                    self.task_queue.put(cross_val_group)
                    tasks_running += 1
                    #print('tasks running: {0} {1}'.format(tasks_running, configuration))
                except Exception as e:
                    if self.console_queue:
                        self.console_queue.put("{0} Exception: {1}\n".format(now(), e))
                    else:
                        sys.stderr.write("{0} Exception: {1}".format(now(), e))
                        #print("{0} Exception: {1}".format(now(), e))
                self.lock.release()
                configuration = self.optimizer.get_next_configuration()
                if not self.result_queue.empty() or tasks_running >= self.max_tasks:
                    try:
                        job_results = self.result_queue.get(timeout=(self.finish_at - datetime.now()).total_seconds())  # block=tasks_running >= self.max_tasks)
                    except Empty:
                        job_results = None
                    if job_results:
                        self.notify_results(job_results)  # GaussianProcessClassifier fails here
                    tasks_running -= 1
                command = None
                try:
                    command = self.command_queue.get_nowait()
                except Empty:
                    command = None
                if command == 'PAUSE':
                    command = None
                    try:
                        command = self.command_queue.get()
                    except Exception:
                        print("error reading from command queue")
                if command == 'STOP':
                    self.task_queue.put(0)
                    self.finish_at = datetime.now()

                if self.timeout:
                    configuration = None
                    #print('! tasks running: {0}'.format(tasks_running))
                # Report completion to job manager
                #configuration = None
                import time
                #time.sleep(10)
        except KeyboardInterrupt:
            print("Failing gracefully")
        self.lock.acquire()
        self.task_queue.put((self.learner.__name__, 'Finished'))
        #print(self.learner.__name__, "ended, closing...")
        self.lock.release()
        #print("All tasks sent. {0} waiting for poison pill".format(self.learner.__name__))
        job_results = self.result_queue.get()
        if self.timeout:
            job_results = None  # TODO notify that no more jobs will be returned
        while job_results:
            self.notify_results(job_results)
            job_results = self.result_queue.get()
        self.status = 'Finished'
        if self.console_queue:
            self.console_queue.put("{0} [Task {1} finished. Should exit now ]\n".format(now(), self.learner.__name__))
        else:
            print("{0} [Task {1} finished. Should exit now ]".format(now(), self.learner.__name__))

    def notify_results(self, job):
        if any([issubclass(type(labels), Exception) for labels in job.fold_labels]):
            #print("Exception happened")
            self.lock.acquire()
            self.manager.add_task_results(self, False)
            metrics = ClassificationMetrics()
            evaluation_results = EvaluationResults(self.learner.__name__,
                                                   job.parameters, metrics)
            self.manager.console_queue.put(evaluation_results)
            self.lock.release()
        else:
            if self.manager.dataset.is_regression:
                metrics = RegressionMetrics(self.manager.dataset.get_target(), job.labels, baseline=self.manager.baseline_metrics)
            else:
                try:
                    metrics = ClassificationMetrics(self.manager.dataset.get_target(), job.labels, job.dataset['target_names'], baseline=self.manager.baseline_metrics)
                except ValueError:
                    print("NO ERRORS SHOULD HAPPEN HERE! PLEASE CHECK THIS CODE AGAIN")
                    metrics = ClassificationMetrics()
            evaluation_results = EvaluationResults(self.learner.__name__,
                                                   job.parameters, metrics)
            self.optimizer.add_results(evaluation_results)
            self.lock.acquire()
            #self.manager.console_queue.put("{0}: {1}\n".format(evaluation_results.learner, evaluation_results.metrics.score))
            if self.manager.console_queue:
                self.manager.console_queue.put(evaluation_results)
            else:
                pass  # TODO process messages
            self.manager.add_task_results(self, True)
            self.lock.release()


class SuggestionTaskManager():
    def __init__(self, dataset, learners, parameters, metrics, time, report_exit_caller, console_queue=None, command_queue=None):
        self.dataset = dataset
        self.metrics = metrics
        self.console_queue = console_queue
        self.time = time
        finish_at = timedelta(minutes=self.time) + datetime.now()
        self.lock = Lock()
        self.report_exit_caller = report_exit_caller
        self.baseline_metrics = self.get_baseline_metrics()
        self.proc_manager = Manager()
        self.statuses = self.proc_manager.dict({learner.__name__: False for learner in learners})
        self.ranking = self.proc_manager.dict({learner.__name__: None for learner in learners})
        self.finished = self.proc_manager.dict({learner.__name__: False for learner in learners})
        self.task_queue = Queue()
        self.queues = {learner.__name__: Queue() for learner in learners}
        self.command_queues = {learner.__name__: Queue() for learner in learners}
        self.command_queue = command_queue
        self.suggestion_tasks = {learner.__name__:
                                 SuggestionTask(self, learner, parameters[learner.__name__],
                                                self.task_queue, self.queues[learner.__name__], self.lock, self.console_queue, finish_at, 2,
                                                self.command_queues[learner.__name__])
                                 for learner in learners}
        self.job_manager = JobManager(self.task_queue, self.queues, self.lock, self.finished, self.console_queue)

    def get_baseline_metrics(self, baseline_classifier=BaselineClassifier):
        import numpy as np
        learner = BaselineRegressor() if self.dataset.is_regression else BaselineClassifier()
        cross_val_group = CrossValidationGroup(learner, None, self.dataset)
        folds = cross_val_group.create_folds()
        prediction = []
        for fold in folds:
            learner.train(fold.training_set)
            fold_prediction = learner.predict(fold.testing_set)
            prediction.append(fold_prediction)
        if prediction[0].ndim > 1:
            prediction = np.vstack(prediction)
        else:
            prediction = np.hstack(prediction)
        if self.dataset.is_regression:
            baseline_metrics = RegressionMetrics(self.dataset.get_target(), prediction, standarize=False)
        else:
            baseline_metrics = ClassificationMetrics(self.dataset.get_target(), prediction, self.dataset.target_names, standarize=False)
        return baseline_metrics

    def all_tasks_finished(self):
        #return all(self.statuses.values())
        return all([task.status == 'Finished' for task in self.suggestion_tasks.values()])

    def run_tasks(self, wait=True):
        self.job_manager.daemon = True
        self.job_manager.start()
        self.poison_pill_sent = False
        for suggestion_task in self.suggestion_tasks.values():
            suggestion_task.daemon = True
            suggestion_task.start()

        command_dispatcher = CommandDispatcher(self.command_queue, self.command_queues)
        command_dispatcher.daemon = True
        command_dispatcher.start()
        if wait:
            for suggestion_task in self.suggestion_tasks.values():
                if suggestion_task.is_alive():
                    suggestion_task.join()
                print("{0} [Task Manager] [Waiting for task to finish {1}]".format(now(), suggestion_task.learner.__name__))
                #suggestion_task.join()
                suggestion_task.result_queue.close()
                suggestion_task.result_queue.join_thread()
                print("{0} [Task Manager] [Task {1} finished]".format(now(), suggestion_task.learner.__name__))

            command_dispatcher.join()
            # job_manager only joins after all tasks have finished
            self.job_manager.join()
            self.report_exit_caller(self.ranking)

    def close_finished_task(self, task):
        if task.status == 'Finished':
            task.join()

    def send_rank(self):
        print([len(suggestion_task.optimizer.evaluation_results) for suggestion_task in self.suggestion_tasks.values()])
        '''
        subranking = {learner: self.ranking[learner][:5] for learner in self.ranking.keys() if self.ranking[learner] is not None}
        print([(learner, [result.metrics.score for result in ranking]) for learner, ranking in subranking.iteritems()])
        self.console_queue.put(subranking)
        '''

        #print([len(result) for result in subranking.values()])
        #best_configs = BaseOptimizer.get_best_configs([task.optimizer for task in self.suggestion_tasks.values()])
        #print("\n", str(subranking))
        #print("Best configs: {0}\n{1}".format(best_configs, self.suggestion_tasks.values()))
        # compare with previous?
        #self.console_queue.put(best_configs)

    def add_task_results(self, task, status):
        #self.lock.acquire()
        suggestion_task_name = task.learner.__name__
        self.statuses[suggestion_task_name] = status
        task.optimizer.evaluation_results.reverse()
        self.ranking[suggestion_task_name] = task.optimizer.evaluation_results
        if self.all_tasks_finished():
            if self.console_queue:
                self.console_queue.put("[Task Manager] [ Sending poison pill to job manager ] {0}\n".format(now()))
            else:
                print("[Task Manager] [ Sending poison pill to job manager ] {0}".format(now()))
            self.task_queue.put(None)
            self.poison_pill_sent = True
        #self.lock.release()
