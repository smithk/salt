"""
The :mod:`salt.jobs.base` module provides classes to manage and dispatch
asynchronous tasks.
"""
from multiprocessing import Process
from ..utils.strings import now
import os

# TODO To read dataset fragment by cross-validation group and fold id only,
# make run generate an exception containing its ip; then catch the exception on
# the JobManager to create a new pp.server bound to the specific node only, to
# send the data. Use marshal.dump(obj, file(path, 'w')) and
# marshal.load(file(path))


def run(job, training_set, testing_set, verbose=False):
    '''Run the learning and predicting steps on a given job.
        This function is passed to the pp nodes for remote execution
    '''
    from salt.utils.strings import now as Now  # Import statements must be included explicitly here
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
            print("{0} [{1} fold] Exception: {2} {3}".format(Now(), job.learner.__name__, e, job.parameters))
        job.exception = e
    except KeyboardInterrupt:
        print("Keyboard interruption detected. Task failing gracefully")
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
        This function is passed to the pp nodes for remote execution
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
            print("{0} [{1} fold] Exception: {2} {3}".format(timestamp, job.learner.__name__, e, job.parameters))
        job.exception = e
    except KeyboardInterrupt:
        print("Keyboard interruption detected. Task failing gracefully")
    return job


class LearningJob(object):
    '''Structure to be passed to the remote task, with information about one fold.'''
    def __init__(self, learner, parameters, group_id, repetition, fold_id):
    #def __init__(self, learner, parameters, training_set, testing_set, group_id, fold_id):
        self.learner = learner
        self.parameters = parameters
        #self.training_set = training_set
        #self.testing_set = testing_set
        self.prediction = None  # To be filled in by the learner
        self.group_id = group_id  # TODO Read remotely the required data
        self.repetition = repetition
        self.fold_id = fold_id
        self.finished = False
        self.exception = None
        self.runtime = 0


class JobManager(Process):
    '''Manage and dispatch distributed tasks.'''

    def __init__(self, dataset, task_queue, queues, lock, finished, console_queue, local_cores='autodetect', node_list=('127.0.0.1', )):
        self.job_groups = {}
        self.task_queue = task_queue
        self.console_queue = console_queue
        self.queues = queues
        self.lock = lock
        self.cluster = None
        #self.jobs = None
        self.finished = finished
        self.node_list = tuple(node_list)
        self.local_cores = local_cores if type(local_cores) is int else 'autodetect'
        self.dataset = dataset
        #self.retry_jobs = []
        super(JobManager, self).__init__(target=self.run)

    def run(self):
        import pp
        print("[Job Manager] Started with pid={0}".format(os.getpid()))
        self.cluster = pp.Server(self.local_cores, ppservers=self.node_list, restart=False, socket_timeout=300)
        #self.jobs = {}
        try:
            message = self.task_queue.get()  # Wait until a message arrives
            print(message)
            while message:  # or len(self.retry_jobs) > 0:
                #while len(self.retry_jobs) > 0:
                #    learning_job = self.retry_jobs.pop()
                #    self.send_job(learning_job, 0)
                if type(message) is tuple:
                    task_name, task_signal = message
                    if task_signal == 'Finished':
                        self.cluster.wait()
                        pass
                    #    self.finished[task_name] = True
                    #if all(self.finished.values()):
                    #    message = None  #
                    #    break
                elif type(message) is str:
                    print("JOB MANAGER SHUTTING DOWN")  # Doesn't work all the time
                    wait = False
                    if wait:
                        self.cluster.wait()
                    self.cluster.destroy()
                    message = None
                    wait = False
                    break
                else:
                    self.load_jobs(message)  # Message is an actual task
                    #s = self.cluster.get_stats()
                    #print([t.rworker.__dict__ for t in s.values()])
                import psutil
                proc = psutil.Process(os.getpid())
                memory_percent = proc.get_memory_percent()
                #print("{0:.2f}% memory used".format(memory_percent))
                if memory_percent < 60:
                    #self.lock.acquire()
                    message = self.task_queue.get()
                    #print("processing {0}".format(message))
                    #self.lock.release()
                else:
                    print("Abnormal memory consumption. No more tasks will be run.")
                    self.cluster.destroy()
                    message = None
        except KeyboardInterrupt:
            print("JobManager failing gracefully")

        if self.console_queue:
            self.console_queue.put("{0} [Job Manager] [ All jobs finished ]\n".format(now()))
            self.console_queue.put(1)  # TODO: Change end signal
        else:
            print("{0} [Job Manager] [ All jobs finished ]".format(now()))
        print('')
        self.cluster.print_stats()
        return 0

    def join(self):
        print("[Job Manager] [Exiting...]")
        super(JobManager, self).join()

    def load_jobs(self, cross_validation_group):
        learner_name = cross_validation_group.learner.__name__
        self.finished[learner_name] = False
        if not learner_name in self.job_groups:
            self.job_groups[learner_name] = {}

        folds = cross_validation_group.create_folds()
        group_id = id(cross_validation_group)
        self.job_groups[learner_name][group_id] = cross_validation_group
        fold_num = 1
        #print("{0} [Job Manager] Submitting {1}".format(now(), cross_validation_group))
        if self.console_queue:
            self.console_queue.put("{0} [Job Manager] Submitting {1}\n".format(now(), cross_validation_group))
        else:
            print("{0} [Job Manager] Submitting {1}".format(now(), cross_validation_group))
        for fold in folds:
            learning_job = LearningJob(fold.learner, fold.parameters,
                                       #fold.training_set, fold.testing_set,
                                       group_id=group_id,
                                       repetition=cross_validation_group.repetition,
                                       fold_id=fold_num)
            try:
                #job = self.send_job(learning_job, group_id)
                job = self.cluster.submit(run, (learning_job, fold.training_set, fold.testing_set), modules=('salt.data',), callback=self.notify_status, group=group_id)
                #del job
                #print("{0} [Job Manager]     {1} fold sent ({2}, {3}/{4})".format(now(), learner_name, group_id, fold_num, len(folds)))
                #self.jobs[(group_id, fold_num)] = job
                #import sys
                #print("len {0}".format([sys.getsizeof(group) for group in self.job_groups.values()]))
            except Exception as e:
                print("[Job Manager] [Exception] {0}".format(e))
            fold_num += 1

    def send_job(self, learning_job, group_id='default'):
        job = self.cluster.submit(run_new, (learning_job, id(self.dataset)), modules=('salt.data',), callback=self.notify_status, group=group_id)
        return job

    def send_data(self, ip_address, learning_job):
        try:
            import pp
            ip_address = "{0}:{1}".format(ip_address, 60001)
            cluster = pp.Server(0, ppservers=(ip_address,))
            print("getting fold data...")
            testing_set, training_set = self.dataset.get_fold_data(learning_job.repetition, learning_job.fold_id - 1)
            job = cluster.submit(write_file, (training_set, testing_set, id(self.dataset), learning_job),
                                 modules=('salt.data', 'salt.data.data', 'numpy', 'numpy.core.multiarray'))
            #                     callback=self.send_job)
            job_result = job()
            if type(job_result) is LearningJob:
                self.send_job(job_result)
            #print("received {0}".format(a))
            #cluster.destroy()
            print("data sent to {0}".format(ip_address))
        except Exception as sss:
            print("{0} happened!!! o_O".format(sss))

    def notify_status(self, learning_job):
        import gc
        if learning_job is None:
            print("JOB CRASHED!!!")
        else:
            learner_name = learning_job.learner.__name__
            cross_validation_group = self.job_groups[learner_name][learning_job.group_id]
            cross_validation_group.runtime = learning_job.runtime
            if learning_job.exception is not None:
                if isinstance(learning_job.exception, Exception):
                    print("exception in {1}: {0}".format(learning_job.exception, learning_job.learner.__name__))
                    cross_validation_group.fold_labels[learning_job.fold_id - 1] = learning_job.exception
                else:
                    print("data not found, sending data for {0}, {1}".format(learning_job.repetition, learning_job.fold_id))
                    ip_address = learning_job.exception
                    self.send_data(ip_address, learning_job)
                    learning_job.exception = None
            else:
                cross_validation_group.fold_labels[learning_job.fold_id - 1] = learning_job.prediction
            if all(labels is not None for labels in cross_validation_group.fold_labels):
                result_queue = self.queues[learner_name]
                #print("sending result for {0}".format(cross_validation_group.parameters))
                result_queue.put(cross_validation_group)
                #del self.job_groups[learner_name][learning_job.group_id]
                #del cross_validation_group
                #print("result sent")
                if self.finished[learner_name]:  # finished sending new jobs
                    all_jobs_finished = all(labels is not None for job in self.job_groups[learner_name].values() for labels in job.fold_labels)
                    if all_jobs_finished:
                        #print("{0} [Job Manager] [ All jobs for {1} have finished. Sending poison pill ]".format(now(), learner_name))
                        self.console_queue.put("{0} [Job Manager] [ All jobs for {1} have finished. Sending poison pill ]\n".format(now(), learner_name))
                        result_queue.put(None)
            learning_job.prediction = None
        #del learning_job
                #self.job_groups[learner_name][learning_job.group_id].dataset.DATA = None
        gc.collect()
