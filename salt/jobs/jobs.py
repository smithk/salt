"""
The :mod:`salt.jobs.base` module provides classes to manage and dispatch
asynchronous tasks.
"""
from multiprocessing import Process
from ..utils.strings import now


def run(job):
    from salt.utils.strings import now
    import sys
    from time import time
    try:
        time_0 = time()
        learner = job.learner(**job.parameters)
        #print("training {1} learner with parameters {0}".format(job.parameters, job.fold_id))
        learner.train(job.training_set)
        #print("predicting learner with parameters {0}".format(job.parameters))
        prediction = learner.predict(job.testing_set)
        #print("done {0}".format(job.parameters))
        time_f = time()
        job.runtime = time_f - time_0
        job.prediction = prediction
    except Exception as e:
        print("{0} [{1} fold] Exception: {2} {3}".format(now(), job.learner.__name__, e, job.parameters))
        #sys.stderr.write("{0} [{1} fold] Exception: {2} {3}".format(now(), job.learner.__name__, e, job.parameters))
        # TODO: Report exceptions
        job.exception = e
    except KeyboardInterrupt:
        print("Task failing gracefully")
    return job
    #notify_status(self)


def notify_status(obj):
    print("notifying status")


class LearningJob(object):
    def __init__(self, learner, parameters, training_set, testing_set, group_id, fold_id):
        self.learner = learner
        self.parameters = parameters
        self.training_set = training_set
        self.testing_set = testing_set
        self.prediction = None
        self.group_id = group_id
        self.fold_id = fold_id
        self.finished = False
        self.exception = None
        self.runtime = 0
        #self.job_id = LearningJobManager.get_job_id()

    #def run(self, notify_status):
    #    learner = self.learner(**self.parameters)
    #    learner.train(self.training_set)
    #    prediction = learner.predict(self.testing_set)
    #    self.prediction = prediction
    #    notify_status(self)


class LearningJobManager(object):
    jobs = 0

    def __init__(self):
        self.job_groups = {}

    def load_jobs(self, cross_validation_group, notify_result):
        folds = cross_validation_group.create_folds()
        group_id = id(cross_validation_group)
        self.job_groups[group_id] = cross_validation_group
        self.notify_result = notify_result
        fold_num = 1
        for fold in folds:
            learning_job = LearningJob(fold.learner, fold.parameters,
                                       fold.training_set, fold.testing_set,
                                       group_id=group_id, fold_id=fold_num)
            learning_job.run(self.notify_status)
            fold_num += 1

    def notify_status(self, learning_job):
        # TODO: check exit status
        cross_validation_group = self.job_groups[learning_job.group_id]
        cross_validation_group.fold_labels[learning_job.fold_id - 1] = learning_job.prediction
        print("result ", learning_job.runtime)
        if all(labels is not None for labels in cross_validation_group.fold_labels):
            self.notify_result(cross_validation_group)

    @classmethod
    def get_job_id(self):
        self.jobs += 1
        return self.jobs


class JobManager(Process):
    jobs = 0

    def __init__(self, task_queue, queues, lock, finished, console_queue, local_cores='autodetect', node_list=('127.0.0.1', )):
        self.job_groups = {}
        self.task_queue = task_queue
        self.console_queue = console_queue
        self.queues = queues
        self.lock = lock
        super(JobManager, self).__init__(target=self.run)
        self.cluster = None
        self.jobs = None
        self.finished = finished
        self.node_list = tuple(node_list)
        self.local_cores = local_cores if type(local_cores) is int else 'autodetect'
        #self.cluster = pp.Server(1, ppservers=('10.2.172.4', '10.2.164.194',))
        #self.cluster = JobCluster(run2, callback=notify_status, reentrant=True,
        #                          #ip_addr='10.2.172.4',
        #                          secret='salt',
        #                          loglevel=logging.DEBUG)

    def all_tasks_done(self):
        return all(queue is None for queue in self.queues.values())

    def run(self):
        import pp
        # Decide on an appropriate timeout
        self.cluster = pp.Server(self.local_cores, ppservers=self.node_list, restart=False)
        self.jobs = {}
        #print("{n} cpus available".format(n=self.cluster.get_ncpus()))
        try:
            message = self.task_queue.get()
            while message:
                # print("[Job Manager] [MESSAGE]  {0}".format(message))
                if type(message) is tuple:
                    task_name, task_signal = message
                    if task_signal == 'Finished':
                        self.finished[task_name] = True
                    if all(self.finished.values()):
                        message = None
                    else:
                        message = self.task_queue.get()
                        #self.queues[task_name]
                elif type(message) is int:
                    print("dying!")  # DOESN'T WORK
                    self.cluster.destroy()
                else:
                    # message is request to process a CrossValidationGroup
                    self.load_jobs(message)
                    message = self.task_queue.get()
            #self.cluster.print_stats()
            #for queue in self.queues.values():
                # poison pill to SuggestionTasks
            #    queue.put(None)
            #self.task_queue.close()
            for job in self.jobs.values():
                if not job.finished:
                    pass
                    #self.lock.acquire()
                    #print("waiting for job {0}".format(job.tid))
                    #print("waiting for job {0},{1}".format(result.group_id, result.fold_id))
                    #result = job()
                    #print("job finished")
                    #self.lock.release()
        except KeyboardInterrupt:
            print("JobManager failing gracefully")

        self.cluster.wait()
        for job in self.jobs.values():
            if not job.finished:
                if self.console_queue:
                    self.console_queue.put("{0} [Job Manager] ATTENTION: PROCESSES EXITED PREMATURELY\n".format(now()))
                else:
                    print("{0} [Job Manager] ATTENTION: PROCESSES EXITED PREMATURELY".format(now()))
        if self.console_queue:
            self.console_queue.put("{0} [Job Manager] [ All jobs finished ]\n".format(now()))
            self.console_queue.put(1)  # TODO: Change end signal
        else:
            print("{0} [Job Manager] [ All jobs finished ]".format(now()))
        print('')
        self.cluster.print_stats()

    #def join(self):
    #    print("[Job Manager] [Exiting...]")
    #    super(JobManager, self).join()

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
                                       fold.training_set, fold.testing_set,
                                       group_id=group_id, fold_id=fold_num)
            # No, put only cross validation groups in the list!
            #self.pending_jobs[fold.learner.__name__].append((group_id, fold_num))
            try:
                #job = self.cluster.submit(learning_job)
                job = self.cluster.submit(run, (learning_job,), callback=self.notify_status, group=group_id)
                #print("{0} [Job Manager]     {1} fold sent ({2}, {3}/{4})".format(now(), learner_name, group_id, fold_num, len(folds)))
                self.jobs[(group_id, fold_num)] = job
            except Exception as e:
                print("[Job Manager] [Exception] {0}".format(e))
            #learning_job.run(self.notify_status)
            fold_num += 1

    def notify_status(self, learning_job):
        if learning_job is None:
            print("JOB CRASHED!!!")
        else:
            learner_name = learning_job.learner.__name__
            cross_validation_group = self.job_groups[learner_name][learning_job.group_id]
            cross_validation_group.runtime = learning_job.runtime
            if learning_job.exception is not None:
                # print("exception {0}".format(learning_job.exception.args))
                cross_validation_group.fold_labels[learning_job.fold_id - 1] = learning_job.exception
            else:
                cross_validation_group.fold_labels[learning_job.fold_id - 1] = learning_job.prediction
            if all(labels is not None for labels in cross_validation_group.fold_labels):
                result_queue = self.queues[learner_name]
                #print("sending result for {0}".format(cross_validation_group.parameters))
                result_queue.put(cross_validation_group)
                #print("result sent")
                if self.finished[learner_name]:  # finished sending new jobs
                    all_jobs_finished = all(labels is not None for job in self.job_groups[learner_name].values() for labels in job.fold_labels)
                    if all_jobs_finished:
                        #print("{0} [Job Manager] [ All jobs for {1} have finished. Sending poison pill ]".format(now(), learner_name))
                        self.console_queue.put("{0} [Job Manager] [ All jobs for {1} have finished. Sending poison pill ]\n".format(now(), learner_name))
                        result_queue.put(None)

    @classmethod
    def get_job_id(self):
        self.jobs += 1
        return self.jobs
