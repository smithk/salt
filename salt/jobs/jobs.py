"""
The :mod:`salt.jobs.base` module provides classes to manage and dispatch
asynchronous tasks.
"""
from multiprocessing import Process, Pool
from multiprocessing.pool import ApplyResult
from ..utils.strings import now
from ..data import PredictionSet
from six import iteritems
from dispy import JobCluster, DispyJob
from collections import Mapping
import numpy as np
import os
import gc
import setproctitle

# TODO To read dataset fragment by cross-validation group and fold id only,
# make run generate an exception containing its ip; then catch the exception on
# the JobManager to create a new pp.server bound to the specific node only, to
# send the data. Use marshal.dump(obj, file(path, 'w')) and
# marshal.load(file(path))


def run(job, training_set, training_target, testing_set, verbose=False):
    '''Run the learning and predicting steps on a given job.
        This function is passed to the cluster nodes for remote execution
    '''
    import setproctitle
    setproctitle.setproctitle('SALT task...')
    from salt.utils.strings import now as Now  # Import statements must be included explicitly here
    from salt.learn import AVAILABLE_CLASSIFIERS
    from time import time
    try:
        time_0 = time()
        learner = AVAILABLE_CLASSIFIERS[job.learner](**job.configuration)
        if verbose:
            print("training {1},{2} learner with parameters {0}".format(job.parameters, job.group_id, job.fold))
        learner.train(training_set, training_target)
        if verbose:
            print("predicting {1},{2} learner with parameters {0}".format(job.parameters, job.group_id, job.fold))
        prediction = learner.predict(testing_set)
        time_f = time()
        job.runtime = time_f - time_0
        job.prediction = prediction
    except Exception as e:
        if verbose:
            print("{0} [{1} fold] Exception: {2} {3}".format(Now(), job.learner, e, job.parameters))
        job.exception = e
    except KeyboardInterrupt:
        print("Keyboard interruption detected. Task failing gracefully")
    except:
        print("something else happened")
    return job


def write_file(training_set, testing_set, dataset_id, learning_job, default_cache_path='/dev/shm'):
    from os.path import join
    import cPickle
    writing_result = learning_job  # if learning_job is returned, writing was successful and resending should take place
    try:
        filename = "{0}_{1}_{2}.chunk".format(dataset_id, learning_job.repetition, learning_job.fold_id)
        full_path = join(default_cache_path, filename)
        with open(full_path, 'w') as chunk_file:
            cPickle.dump((training_set, testing_set), chunk_file)
    except Exception as file_not_written:
        print("Exception dumping file into remote node filesystem: {0}".format(file_not_written))
        writing_result = file_not_written
    return writing_result


def run_new(job, dataset_id, verbose=False, default_cache_path='/dev/shm'):
    '''Run the learning and predicting steps on a given job.
        This function is passed to the cluster nodes for remote execution
    '''
    from os.path import join, exists
    import cPickle
    training_set, testing_set = None, None
    file_not_read = None
    try:
        filename = "{0}_{1}_{2}.chunk".format(dataset_id, job.repetition, job.fold_id)
        full_path = join(default_cache_path, filename)
        if exists(full_path):
            with open(full_path) as chunk_file:
                training_set, testing_set = cPickle.load(chunk_file)
        else:
            # ATTENTION: this only works for EC2 nodes
            import subprocess
            proc = subprocess.Popen('curl http://169.254.169.254/latest/meta-data/public-ipv4', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            ip_address = proc.stdout.read()
            file_not_read = ip_address
    except Exception as file_not_read:
        print("Exception loading dataset on remote node: {0}".format(file_not_read))
    if training_set is None or testing_set is None:
        # Send request for file contents
        job.exception = file_not_read
        return job

    from time import time
    try:
        time_0 = time()
        learner = job.learner(**job.parameters)
        if verbose:
            print("training {1},{2} learner with parameters {0}".format(job.parameters, job.group_id, job.fold_id))
        learner.train(training_set)
        if verbose:
            print("predicting {1},{2} learner with parameters {0}".format(job.parameters, job.group_id, job.fold_id))
        prediction = learner.predict(testing_set)
        time_f = time()
        job.runtime = time_f - time_0
        job.prediction = prediction
    except Exception as e:
        if verbose:
            timestamp = time.strftime("<%T>")
            print("{0} [{1} fold] Exception: {2} {3}".format(timestamp, job.learner, e, job.parameters))
        job.exception = e
    except KeyboardInterrupt:
        print("Keyboard interruption detected. Task failing gracefully")
    return job


class LearningJob(object):
    '''Structure to be passed to the remote task, with information about one fold.'''
    def __init__(self, learner, configuration, repetition, fold):
    #def __init__(self, learner, parameters, training_set, testing_set, group_id, fold_id):
        self.learner = learner
        self.configuration = configuration
        self.repetition = repetition
        self.fold = fold

        # To be filled in by the learner
        self.prediction = None
        self.runtime = np.inf
        self.exception = None


class DistributedJobManager(Process):
    '''Manage and dispatch distributed tasks.'''

    def __init__(self, dataset, task_queue, queues, lock, finished, console_queue, local_cores='autodetect', node_list=('127.0.0.1', ), ip_addr='127.0.0.1'):
        self.name = 'DistJobManager'
        self.job_groups = {}
        self.task_queue = task_queue
        self.console_queue = console_queue
        self.queues = queues
        self.lock = lock
        self.cluster = None
        #self.jobs = None
        self.finished = finished
        self.node_list = node_list  # tuple(node_list) pp requires tuple
        self.local_cores = local_cores if type(local_cores) is int else 'autodetect'
        self.dataset = dataset
        self.ip_addr = ip_addr
        #self.retry_jobs = []

        self.active_jobs = {}
        self.active_configurations = {}

        super(DistributedJobManager, self).__init__(target=self.run)

    def get_used_memory(self):
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_percent()

    def run(self, wait=False):
        '''
        Listen to messages and send jobs.

        Parameters
        ----------
            wait : bool, optional
                   Wait for all jobs to finish before closing the JobManager.
        '''
        print("[Job Manager] Started with pid={0}".format(os.getpid()))
        cluster_args = {'nodes': self.node_list,  # 'port': 51348,
                        'pulse_interval': 60,
                        'ping_interval': 60,
                        'poll_interval': 5,
                        'reentrant': True,
                        'ip_addr': self.ip_addr,
                        #'loglevel': logging.DEBUG,
                        'callback': self.notify_status}
        self.cluster = JobCluster(run, **cluster_args)
        try:
            message = self.task_queue.get()  # Wait until a message arrives
            while message:
                if type(message) is str:
                    print("JOB MANAGER SHUTTING DOWN")  # Doesn't work all the time
                    if wait:
                        self.cluster.wait()
                    else:
                        for learner, job_list in iteritems(self.active_jobs):
                            for configuration, jobs in iteritems(job_list):
                                for job in jobs:
                                    if type(job) is DispyJob:
                                        if job.status in (DispyJob.Created, DispyJob.Running):
                                            self.cluster.cancel(job)
                    message = None
                elif type(message) is tuple:
                    learner, message_body = message
                    if isinstance(message_body, Mapping):
                        self.load_jobs(*message)  # Message is an actual task (learner, config)
                    else:
                        if message_body == 'Finished':
                            pass
                # TODO: receive status request, send status self.cluster.stats()
                else:
                    print("MESSAGE UNKNOWN {0}".format(message))

                memory_percent = self.get_used_memory()  # TODO Do this less often
                if memory_percent < 60:
                    message = self.task_queue.get()  # Continue accepting messages
                else:
                    # TODO Free memory instead of stop
                    print("Abnormal memory consumption. No more tasks will be run.")
                    self.cluster.wait()
                    message = None
            print("Job Manager finished. Joining the main process")
        except KeyboardInterrupt:
            print("Ctrl+C detected. JobManager failing gracefully (?)")
        except:
            print("problem here!")

        if self.console_queue is not None:
            self.console_queue.put("{0} [Job Manager] [ All jobs finished ]\n".format(now()))
            self.console_queue.put(1)  # TODO: Change end signal and don't send text
        else:
            print("{0} [Job Manager] [ All jobs finished ]".format(now()))
        print(self.cluster.stats())
        self.cluster.close()
        return 0

    def join(self):
        super(DistributedJobManager, self).join()
        self.cluster.close()
        self.cluster.join()
        print("[Job Manager] [Exiting...]")

    def load_jobs(self, learner, configuration):
        message = "{0} [Job Manager] Submitting {1}:{2}".format(now(), learner, configuration)
        if self.console_queue is not None:
            self.console_queue.put(message)
        else:
            print(message)
        # Create prediction set
        prediction_set = PredictionSet(learner, configuration,
                                       self.dataset.repetitions, self.dataset.folds)
        learner_prediction = self.active_configurations.get(learner)
        if learner_prediction is None:
            learner_prediction = {}
            self.active_configurations[learner] = learner_prediction
        learner_prediction[self._key(configuration)] = prediction_set

        # Create active job list
        active_learner_jobs = self.active_jobs.get(learner)
        if active_learner_jobs is None:
            active_learner_jobs = {}
            self.active_jobs[learner] = active_learner_jobs
        active_configuration_jobs = []
        active_learner_jobs[self._key(configuration)] = active_configuration_jobs

        success_sending_jobs = True

        for repetition in xrange(self.dataset.repetitions):
            for fold in xrange(self.dataset.folds):
                learning_job = LearningJob(learner, configuration, repetition, fold)
                testing_set, training_set = self.dataset.get_fold_data(repetition, fold)
                training_data, training_labels = training_set
                testing_data, testing_labels = testing_set
                try:
                    job = self.cluster.submit(learning_job, training_data, training_labels, testing_data)
                    active_configuration_jobs.append(job)
                except Exception as e:
                    success_sending_jobs = False
                    print("[Job Manager] [Exception sending jobs] {0}".format(e))
        if not success_sending_jobs:
            print("couldn't send jobs")
            # Terminate active jobs for the current configuration, if any
            for active_job in active_configuration_jobs:
                self.cluster.cancel(active_job)
            del active_learner_jobs[self._key(configuration)]
            # Remove pending predictions
            del learner_prediction[self._key(configuration)]

    def _key(self, dictionary):
        return repr(sorted(dictionary.items()))

    def notify_status(self, cluster_job):
        self.lock.acquire()
        learning_job = cluster_job.result
        try:
            if learning_job is not None:  # TODO Also check for exception in cluster_job
                exception = learning_job.exception
                learner = learning_job.learner
                configuration = learning_job.configuration
                repetition = learning_job.repetition
                fold = learning_job.fold
                prediction = learning_job.prediction
                runtime = learning_job.runtime

                # TODO Test if notify_status happens on canceled jobs.
                active_learner_configurations = self.active_configurations[learner]
                prediction_set = active_learner_configurations[self._key(configuration)]

                if exception is not None:
                    if isinstance(exception, Exception):
                        #print("exception in {1}: {0}".format(exception, learner))
                        prediction_set.add(exception, repetition, fold, runtime)
                    else:  # TODO Repair data sending. It doesn't currently work
                        print("data not found, sending data for {0}, {1}".format(repetition, fold))
                        ip_address = exception
                        self.send_data(ip_address, learning_job)
                        exception = None
                else:
                    prediction_set.add(prediction, repetition, fold, runtime)
                if all(prediction is not None for prediction in prediction_set.predictions):
                    result_queue = self.queues[learner]
                    result_queue.put(prediction_set)
                    del self.active_configurations[learner][self._key(configuration)]
                    del self.active_jobs[learner][self._key(configuration)]
                    #print(self.cluster.stats())
                else:
                    pass
            else:
                if cluster_job.exception is not None:
                    print("Job crashed: {0}".format(cluster_job.exception))
        except Exception as e:
            print("[Job Manager] Something is not right: {0}, {1}".format(e, type(e)))
        finally:
            self.lock.release()
        gc.collect()


class LocalJobManager(Process):
    '''Manage and dispatch distributed tasks.'''

    def __init__(self, dataset, task_queue, queues, lock, finished, console_queue, local_cores):
        self.job_groups = {}
        self.task_queue = task_queue
        self.console_queue = console_queue
        self.queues = queues
        self.lock = lock
        self.cluster = None
        #self.jobs = None
        self.finished = finished
        self.local_cores = local_cores if type(local_cores) is int else None
        self.dataset = dataset

        self.active_jobs = {}
        self.active_configurations = {}

        super(LocalJobManager, self).__init__(target=self.run)

    def get_used_memory(self):
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_percent()

    def run(self, wait=False):
        setproctitle.setproctitle("SALT LocalJobManager")
        '''
        Listen to messages and send jobs.

        Parameters
        ----------
            wait : bool, optional
                   Wait for all jobs to finish before closing the JobManager.
        '''
        print("[Job Manager] Started with pid={0}".format(os.getpid()))
        cluster_args = {'processes': self.local_cores}
        self.cluster = Pool(**cluster_args)
        try:
            message = self.task_queue.get()  # Wait until a message arrives
            while message:
                if type(message) is str:
                    print("JOB MANAGER SHUTTING DOWN")  # Doesn't work all the time
                    if wait:
                        self.cluster.close()
                    else:
                        for learner, job_list in iteritems(self.active_jobs):
                            for configuration, jobs in iteritems(job_list):
                                for job in jobs:
                                    if type(job) is ApplyResult:
                                        if not job.ready():
                                            pass  # self.cluster.cancel(job)
                    message = None
                elif type(message) is tuple:
                    learner, message_body = message
                    if isinstance(message_body, Mapping):
                        self.load_jobs(*message)  # Message is an actual task (learner, config)
                    else:
                        if message_body == 'Finished':
                            pass
                else:
                    print("MESSAGE UNKNOWN {0}".format(message))

                memory_percent = self.get_used_memory()  # TODO Do this less often
                if memory_percent < 60:
                    message = self.task_queue.get()  # Continue accepting messages
                else:
                    # TODO Free memory instead of stop
                    print("Abnormal memory consumption. No more tasks will be run.")
                    self.cluster.close()
                    message = None
        except KeyboardInterrupt:
            print("Ctrl+C detected. JobManager failing gracefully (?)")
        except Exception as ex:
            print("Problem here!", ex)

        if self.console_queue is not None:
            self.console_queue.put("{0} [Job Manager] [ All jobs finished ]\n".format(now()))
            self.console_queue.put(1)  # TODO: Change end signal and don't send text
        else:
            print("{0} [Job Manager] [ All jobs finished ]".format(now()))
        self.cluster.close()
        self.cluster.join()
        print("job manager exists here")
        return 0

    #def join(self):
    #    print("[Job Manager] [Exiting...]")
    #    super(LocalJobManager, self).join()

    def load_jobs(self, learner, configuration):
        message = "{0} [Job Manager] Submitting {1}:{2}".format(now(), learner, configuration)
        if self.console_queue is not None:
            self.console_queue.put(message)
        else:
            print(message)
        # Create prediction set
        prediction_set = PredictionSet(learner, configuration,
                                       self.dataset.repetitions, self.dataset.folds)
        learner_prediction = self.active_configurations.get(learner)
        if learner_prediction is None:
            learner_prediction = {}
            self.active_configurations[learner] = learner_prediction
        learner_prediction[self._key(configuration)] = prediction_set

        # Create active job list
        active_learner_jobs = self.active_jobs.get(learner)
        if active_learner_jobs is None:
            active_learner_jobs = {}
            self.active_jobs[learner] = active_learner_jobs
        active_configuration_jobs = []
        active_learner_jobs[self._key(configuration)] = active_configuration_jobs

        success_sending_jobs = True

        for repetition in xrange(self.dataset.repetitions):
            for fold in xrange(self.dataset.folds):
                learning_job = LearningJob(learner, configuration, repetition, fold)
                testing_set, training_set = self.dataset.get_fold_data(repetition, fold)
                training_data, training_labels = training_set
                testing_data, testing_labels = testing_set
                try:
                    job = self.cluster.apply_async(run, (learning_job, training_data, training_labels, testing_data), callback=self.notify_status)
                    active_configuration_jobs.append(job)
                except Exception as e:
                    success_sending_jobs = False
                    print("[Job Manager] [Exception sending jobs] {0}".format(e))
        if not success_sending_jobs:
            print("couldn't send jobs")
            # Terminate active jobs for the current configuration, if any
            for active_job in active_configuration_jobs:
                self.cluster.cancel(active_job)  # TODO cancel jobs in multiprocessing
            del active_learner_jobs[self._key(configuration)]
            # Remove pending predictions
            del learner_prediction[self._key(configuration)]

    def _key(self, dictionary):
        return repr(sorted(dictionary.items()))

    def notify_status(self, cluster_job):
        #self.lock.acquire()
        learning_job = cluster_job  # cluster_job.result
        try:
            if learning_job is not None:  # TODO Also check for exception in cluster_job
                exception = learning_job.exception
                learner = learning_job.learner
                configuration = learning_job.configuration
                repetition = learning_job.repetition
                fold = learning_job.fold
                prediction = learning_job.prediction
                runtime = learning_job.runtime

                # TODO Test if notify_status happens on canceled jobs.
                active_learner_configurations = self.active_configurations[learner]
                prediction_set = active_learner_configurations[self._key(configuration)]

                if exception is not None:
                    if isinstance(exception, Exception):
                        print("exception in {1}: {0}".format(exception, learner))
                        prediction_set.add(exception, repetition, fold, runtime)
                    else:  # TODO Repair data sending. It doesn't currently work
                        print("data not found, sending data for {0}, {1}".format(repetition, fold))
                        ip_address = exception
                        self.send_data(ip_address, learning_job)
                        exception = None
                else:
                    prediction_set.add(prediction, repetition, fold, runtime)
                if all(prediction is not None for prediction in prediction_set.predictions):
                    result_queue = self.queues[learner]
                    result_queue.put_nowait(prediction_set)
                    del self.active_configurations[learner][self._key(configuration)]
                    del self.active_jobs[learner][self._key(configuration)]
                    #print(self.cluster.stats())
                else:
                    pass
            else:
                if cluster_job.exception is not None:
                    print("Job crashed: {0}".format(cluster_job.exception))
        except Exception as e:
            print("Something is not right: {0}, {1}".format(e, type(e)))
        #finally:
        #    self.lock.release()
        gc.collect()
