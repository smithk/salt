"""The :mod:`salt.suggest.base` module provides classes to describe suggestion tasks."""

from Queue import Empty
from multiprocessing import Process, Queue, Lock, Manager
from ..learn.classifiers import BaselineClassifier
from ..learn.regressors import BaselineRegressor
from ..optimize import KDEOptimizer, ShrinkingHypercubeOptimizer, DefaultConfigOptimizer
from ..jobs import JobManager
from ..evaluate.metrics import ClassificationMetrics, RegressionMetrics
from ..evaluate import EvaluationResultSet
from ..utils.strings import now
from datetime import datetime, timedelta
import sys
import numpy as np
import os


class CommandDispatcher(Process):
    '''Broadcast messages to all running processes via the command queue.'''
    def __init__(self, command_queue, process_queues):
        self.command_queue = command_queue
        self.process_queues = process_queues
        super(CommandDispatcher, self).__init__(target=self.run)

    def run(self):
        print("Command dispatcher running. PID={0}".format(os.getpid()))
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
                 task_queue, result_queue, lock, console_queue, finish_at, max_tasks, command_queue, optimizer):
        self.manager = suggestion_task_manager
        self.learner = learner
        self.parameters = parameters
        #self.optimizer = SequentialOptimizer(self.parameters)
        #optimizer_values = ('None (run with default parameters)', 'Random search', 'Shrinking hypercube')
        if optimizer == 'Random search':
            self.optimizer = KDEOptimizer(self.parameters)
        elif optimizer == 'Shrinking hypercube':
            self.optimizer = ShrinkingHypercubeOptimizer(self.parameters)
        elif optimizer == 'None (run with default parameters)':
            self.optimizer = DefaultConfigOptimizer(self.parameters)
        else:
            raise ValueError('Invalid optimization method')
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
        print("{0} [Task] [ {1} finished ]".format(now(), self.learner))

    def _is_timeout(self):
        self._c += 1
        #print(datetime.now(), self.finish_at)
        return datetime.now() > self.finish_at
        #return self._c > 10

    timeout = property(_is_timeout)

    def send_task(self, learner, configuration):
        task_sent = False
        try:
            task = (learner, configuration)
            self.task_queue.put(task)
            task_sent = True
        except Exception as e:
            error_message = "{0} Exception sending task to the cluster: {1}\n".format(now(), e)
            if self.console_queue:
                self.console_queue.put(error_message)
            else:
                sys.stderr.write(error_message)
        return task_sent

    def process_finished_tasks(self):
        '''Check if finished cross-validation groups have been finished and process them.'''
        tasks_processed = 0
        while not self.result_queue.empty() and not self.timeout:  # Ignore results after timeout
            try:
                timeout_seconds = (self.finish_at - datetime.now()).total_seconds()
                job_results = self.result_queue.get(timeout=timeout_seconds)
            except Empty:
                job_results = None
            if job_results:
                self.notify_results(job_results)  # TODO GaussianProcessClassifier fails here. Test why
                tasks_processed += 1
            self.process_commands()
        return tasks_processed

    def process_commands(self):
        '''Process commands if any available.'''
        command = None
        try:
            command = self.command_queue.get_nowait()
        except Empty:
            command = None
        if command == 'PAUSE':
            command = None
            try:
                # Wait for the next command (block the queue meanwhile)
                # TODO Process finished tasks that might be returned by the
                # cluster while paused, to prevent from clogging the result queue.
                command = self.command_queue.get()
            except Exception:
                print("error reading from command queue")
        if command == 'STOP':
            print("{0} Task: STOP SIGNAL ARRIVED".format(self.learner))
            self.task_queue.put('DIE!')  # Poison pill for the job manager
            self.finish_at = datetime.now()  # Force timeout
        if type(command) is datetime:
            self.finish_at = command
            print("new timeout at {0}".format(self.finish_at))

    def run(self):
        print("Learner {0} running. PID={1}".format(self.learner, os.getpid()))
        # Run learner with default parameters
        configuration = self.optimizer.get_next_configuration()
        tasks_running = 0  # Task == running configuration
        try:
            while configuration is not None:
                if tasks_running > self.max_tasks - 1:
                    while tasks_running > self.max_tasks / 2 and not self.timeout:
                        # Wait until cluster processes at least half of its
                        # load to send a new batch
                        tasks_processed = self.process_finished_tasks()
                        tasks_running -= tasks_processed
                        #if tasks_processed > 0:
                        #    print("{0} tasks running".format(tasks_running))
                # Create job batch: r x k folds
                task_sent = self.send_task(self.learner, configuration)
                if task_sent:
                    tasks_running += 1
                else:
                    print("couldn't send new task")
                # Process finished tasks if available
                tasks_processed = self.process_finished_tasks()
                tasks_running -= tasks_processed
                self.process_commands()
                if self.timeout:
                    # TODO Send only one poison pill.
                    self.task_queue.put('DIE!')  # Poison pill for the job manager
                    configuration = None
                else:
                    import time
                    time.sleep(2)
                    configuration = self.optimizer.get_next_configuration()
        except KeyboardInterrupt:
            print("Ctrl+C detected. {0} task shutting down...".format(self.learner))
        self.task_queue.put((self.learner, 'Finished'))
        while tasks_running > 0 and not self.timeout:
            tasks_processed = self.process_finished_tasks()
            tasks_running -= tasks_processed
            if tasks_processed > 0:
                print("{0} tasks running".format(tasks_running))
        #job_results = self.result_queue.get()
        if self.timeout:
            job_results = None  # TODO notify that no more jobs will be returned
            self.notify_results(job_results)
        self.status = 'Finished'
        if self.console_queue:
            self.console_queue.put("{0} [Task {1} finished. Should exit now ]\n".format(now(), self.learner))
        else:
            print("{0} [Task {1} finished. Should exit now ]".format(now(), self.learner))

    def combine_labels(self, fold_labels):
        return np.vstack(fold_labels)

    def notify_results(self, prediction_set):
        if prediction_set is None:
            return
        if any([issubclass(type(labels), Exception) for labels in prediction_set.predictions]):
            # Exception happened
            #self.lock.acquire()
            try:
                self.manager.add_task_results(self, False)
                metrics = ClassificationMetrics()
                #results = EvaluationResults(self.learner,
                #                            prediction_set.configuration, metrics)
                #self.manager.console_queue.put(results)
            except:
                pass
            #finally:
            #    self.lock.release()
        else:
            if self.manager.dataset.is_regression:
                pass  # metrics = RegressionMetrics(self.manager.dataset.get_target(), .labels, baseline=self.manager.baseline_metrics)
            else:
                try:
                    #job_labels = self.combine_labels(job.fold_labels)  # Probabilities
                    fold_metrics = []
                    for repetition in xrange(self.manager.dataset.repetitions):
                        for fold in xrange(self.manager.dataset.folds):
                            fold_target = self.manager.dataset.get_target(repetition, fold)
                            fold_prediction = prediction_set.get(repetition, fold)
                            metrics = ClassificationMetrics(fold_target, fold_prediction,
                                                            self.manager.dataset['target_names'],
                                                            baseline=self.manager.baseline_metrics,
                                                            standardize=True)
                            # ATTENTION: standardize=False doesn't make use of the
                            # baseline!!!! (Deactivated temporarily for testing)

                            fold_metrics.append(metrics)
                except ValueError:
                    print("NO ERRORS SHOULD HAPPEN HERE! PLEASE CHECK THIS CODE AGAIN")
                    fold_metrics = [ClassificationMetrics()]

            evaluation_result_set = EvaluationResultSet(self.learner, prediction_set.configuration, fold_metrics)
            if evaluation_result_set.configuration != {}:
                self.optimizer.add_results(evaluation_result_set)
            if self.manager.console_queue is not None:
                self.manager.console_queue.put(evaluation_result_set)
            else:
                pass
            self.manager.add_task_results(self, True)
            '''
            for metrics in fold_metrics:
                evaluation_results = EvaluationResults(self.learner,
                                                       prediction_set.configuration, metrics)
                print("{0}: {1}\n".format(evaluation_results.learner, evaluation_results.metrics.score))
                if evaluation_results.parameters != {}:
                    #self.lock.acquire()
                    self.optimizer.add_results(evaluation_results)
                    #self.lock.release()
                #self.manager.console_queue.put("{0}: {1}\n".format(evaluation_results.learner, evaluation_results.metrics.score))
                if self.manager.console_queue:
                    self.manager.console_queue.put(evaluation_results)
                else:
                    pass  # TODO process messages
                self.manager.add_task_results(self, True)
            '''


class SuggestionTaskManager():
    def __init__(self, dataset, learners, parameters, metrics, time, report_exit_caller,
                 console_queue=None, command_queue=None, local_cores=0, node_list=None,
                 optimizer='', max_jobs=10, ip_addr='127.0.0.1'):
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
                                 SuggestionTask(self, learner.__name__, parameters[learner.__name__],
                                                self.task_queue, self.queues[learner.__name__], self.lock, self.console_queue, finish_at, max_jobs,
                                                self.command_queues[learner.__name__], optimizer=optimizer)
                                 for learner in learners}
        self.job_manager = JobManager(self.dataset, self.task_queue, self.queues, self.lock,
                                      self.finished, self.console_queue, local_cores, node_list,
                                      ip_addr)

    def get_baseline_metrics(self, baseline_classifier=BaselineClassifier):
        learner = BaselineRegressor() if self.dataset.is_regression else BaselineClassifier()
        learner.train(self.dataset.data, self.dataset.target)
        prediction = learner.predict(self.dataset.data)
        if self.dataset.is_regression:
            baseline_metrics = RegressionMetrics(self.dataset.target, prediction, standardize=False)
        else:
            baseline_metrics = ClassificationMetrics(self.dataset.target, prediction, self.dataset.target_names, standardize=False)
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
            #a = [(result.learner, result.parameters.values(), result.metrics.score) for result in self.optimizer.evaluation_results]
            #with open('/tmp/all_results.txt', 'w') as f:
            #    for i in a:
            #        f.write("{0}, {1}, {2}\n".format(*i))
            self.report_exit_caller(self.ranking)

    def close_finished_task(self, task):
        if task.status == 'Finished':
            task.join()

    def send_rank(self):
        raise NotImplemented
        pass
        #print([len(suggestion_task.optimizer.evaluation_results) for suggestion_task in self.suggestion_tasks.values()])
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
        suggestion_task_name = task.learner
        self.statuses[suggestion_task_name] = status
        task.optimizer.evaluation_results.reverse()
        #self.ranking[suggestion_task_name] = task.optimizer.evaluation_results
        self.ranking[suggestion_task_name] = []
        self.ranking[suggestion_task_name].extend(task.optimizer.evaluation_results)
        if self.all_tasks_finished():
            if self.console_queue:
                self.console_queue.put("[Task Manager] [ Sending poison pill to job manager ] {0}\n".format(now()))
            else:
                print("[Task Manager] [ Sending poison pill to job manager ] {0}".format(now()))
            self.task_queue.put(None)
            self.poison_pill_sent = True
        #self.lock.release()
