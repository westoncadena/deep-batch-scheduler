import numpy as np
import math
import sys
import os
import json

import gym
from gym import spaces
from gym.utils import seeding

from hpc.envs.job import Job
from hpc.envs.job import Workloads
from hpc.envs.cluster import Cluster

# HPC Batch Scheduler Simulation.
# 
# Created by Dong Dai. Licensed on the same terms as the rest of OpenAI Gym.

MAX_QUEUE_SIZE = 63
MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

# each job has features
# submit_time, request_number_of_processors, request_time,
# user_id, group_id, executable_number, queue_number
JOB_FEATURES = 3
MAX_JOBS_EACH_BATCH = 200
DEBUG = False

CANDIDATE_ALGORITHMS = 5

class HpcEnv(gym.Env):

    def __init__(self):  # do nothing and return. A workaround for passing parameters to the environment
        super(HpcEnv, self).__init__()
        self.action_space = spaces.Discrete(CANDIDATE_ALGORITHMS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(JOB_FEATURES * (MAX_QUEUE_SIZE + 1),),
                                            dtype=np.float32)

        self.np_random = self.np_random, seed = seeding.np_random(1)

        self.job_queue = []
        for i in range(0, MAX_QUEUE_SIZE):
            self.job_queue.append(Job())

        self.priority_function = None

        self.running_jobs = []
        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.schedule_logs = []
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.next_arriving_job_idx = 0

        self.Metrics_Queue_Length = 0  # Determine the quality of the job sequence.
        self.Metrics_Probe_Times = 0  # Determine the quality of the job sequence.
        self.Metrics_Total_Execution_Time = 0  # Max(job.scheduled_time + job.run_time)
        self.Metrics_Average_Response_Time = 0  # (job.scheduled_time + job.run_time - job.submit_time) / num_of_jobs
        self.Metrics_Average_Slow_Down = 0  # (job.scheduled_time - job.submit_time) / num_of_jobs
        self.Metrics_System_Utilization = 0  # (cluster.used_node * t_used / cluster.total_node * t_max)

        # slurm
        self.priority_max_age = 0.0
        self.priority_weight_age = 0.0
        self.priority_weight_fair_share = 0.0
        self.priority_favor_small = True
        self.priority_weight_job_size = 0.0
        self.priority_weight_partition = 0.0
        self.priority_weight_qos = 0.0
        self.tres_weight_cpu = 0.0
        self._configure_slurm(1000, 0, 1000, 0, 0, 0, 60 * 60 * 72, True)

        self.scheduler_algs = {
                0: self.fcfs_priority,
                1: self.smallest_job_first,
                2: self.shortest_job_first,
                3: self.largest_job_first,
                4: self.longest_job_first,
        }

        self.pre_processed_job_metrics = {}
        self.pre_processed_job_list = []

        self.loads = None
        self.cluster = None

    def my_init(self, workload_file = '', pre_processed_metrics_file = ''):
        print ("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Ricc", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)

        with open(pre_processed_metrics_file, 'r') as f:
            self.pre_processed_job_metrics = json.load(f)
        self.pre_processed_job_list = list(self.pre_processed_job_metrics)  # get the key list of high quality items

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the space.

        Two things are done in "reset":
        1. randomly choose a batch or a sample of jobs to learn
        2. randomly generate a bunch of jobs that already been scheduled and running to consume random resources.

        @update: need to reconsider "random". if we randomly select the samples, the agent could be confused as the
        rewards are no long consistent and not comparable.
        """
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        for i in range(0, MAX_QUEUE_SIZE):
            self.job_queue.append(Job())

        self.priority_function = None
        self.running_jobs = []
        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.schedule_logs = []
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.next_arriving_job_idx = 0

        self.Metrics_Queue_Length = 0
        self.Metrics_Probe_Times = 0
        self.Metrics_Total_Execution_Time = 0
        self.Metrics_Average_Response_Time = 0
        self.Metrics_Average_Slow_Down = 0
        self.Metrics_System_Utilization = 0
        self._configure_slurm(1000, 0, 1000, 0, 0, 0, 60 * 60 * 72, True)

        # randomly choose a start point in current workload (has to be high quality sequence)
        high_quality_sample_size = len(self.pre_processed_job_metrics)
        self.start = int(self.pre_processed_job_list[self.np_random.randint(0, high_quality_sample_size)])
        # print ("reset, select item: ", self.start)
        # how many jobs are remained in the workload
        job_remainder = self.loads.size() - self.start
        # how many jobs in this batch
        # self.num_job_in_batch = self.np_random.randint(MAX_JOBS_EACH_BATCH, 2 * MAX_JOBS_EACH_BATCH)
        self.num_job_in_batch = MAX_JOBS_EACH_BATCH
        # self.num_job_in_batch = 635
        # the id of the last job in this batch.
        self.last_job_in_batch = self.start + self.num_job_in_batch
        # start scheduling from the first job.
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue[0] = self.loads[self.start]
        # next arriving job would be the second job in the batch
        self.next_arriving_job_idx = self.start + 1

        if DEBUG:
            print("Reset Env: ", self.loads[self.start], ", ", self.num_job_in_batch, ", ", self.loads[
                self.last_job_in_batch])

        # Generate some running jobs to randomly fill the cluster.
        '''
        running_job_size = self.np_random.randint(1, MAX_JOBS_EACH_BATCH)  # size of running jobs.
        for i in range(running_job_size):
            req_num_of_processors = self.np_random.randint(1, self.loads.max_procs) # random number of requests
            runtime_of_job = self.np_random.randint(1, self.loads.max_exec_time)    # random execution time
            job_tmp = Job()
            job_tmp.job_id = (-1 - i) # to be different from the normal jobs; normal jobs have a job_id >= 1
            job_tmp.request_number_of_processors = req_num_of_processors
            job_tmp.run_time = runtime_of_job
            if self.cluster.can_allocated(job_tmp):
                self.running_jobs.append(job_tmp)
                # assume job was randomly generated
                job_tmp.scheduled_time = max(0, (self.current_timestamp - self.np_random.randint(runtime_of_job)))
                if DEBUG:
                    print ("In reset, allocate for job, ", job_tmp, " with free nodes: ", self.cluster.free_node)
                job_tmp.allocated_machines = self.cluster.allocate(job_tmp.job_id, job_tmp.request_number_of_processors)
            else:
                break
        '''
        obs = self.build_observation()
        return obs

    def build_observation(self):
        sq = int(math.ceil(math.sqrt(MAX_QUEUE_SIZE)))
        vector = np.zeros((sq, sq, JOB_FEATURES), dtype=float)
        # self.job_queue.sort(key=lambda j: j.request_number_of_processors)

        for i in range(0, MAX_QUEUE_SIZE):
            job = self.job_queue[i]

            submit_time = job.submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time
            run_time = job.run_time
            # not used for now
            #user_id = job.user_id
            #group_id = job.group_id
            #executable_number = job.executable_number
            #queue_number = job.queue_number

            if job.job_id == 0:
                wait_time = 0
            else:
                wait_time = self.current_timestamp - submit_time
            normalized_wait_time = float(wait_time) / float(MAX_WAIT_TIME)
            normalized_request_time = float(request_time) / float(MAX_RUN_TIME)
            # normalized_run_time = float(run_time) / float(MAX_RUN_TIME)
            normalized_request_nodes = float(request_processors) / float(self.loads.max_procs)

            vector[int(i / sq), int(i % sq)] = [normalized_wait_time, normalized_request_time, normalized_request_nodes]

        cluster_usage = float(self.cluster.free_node) / float(self.cluster.total_node)
        vector[int (MAX_QUEUE_SIZE / sq), int(MAX_QUEUE_SIZE % sq)] = [cluster_usage, cluster_usage, cluster_usage]
        return np.reshape(vector, [-1, (MAX_QUEUE_SIZE + 1) * JOB_FEATURES])

    def _is_job_queue_empty(self):
        return all(v.job_id == 0 for v in self.job_queue)

    def _is_job_queue_full(self):
        return all(v.job_id > 0 for v in self.job_queue)

    def _job_queue_size(self):
        size = 0
        for i in range(0, MAX_QUEUE_SIZE):
            if self.job_queue[i].job_id > 0:
                size += 1
        return size

    def slurm_priority(self, job):
        age_factor = float(job.slurm_age) / float(self.priority_max_age)
        if age_factor > 1:
            age_factor = 1.0
        if self.priority_favor_small:
            size_weight = self.priority_weight_job_size * (1.0 - job.slurm_job_size)
        else:
            size_weight = self.priority_weight_job_size * job.slurm_job_size

        priority_score = \
            self.priority_max_age * age_factor + self.priority_weight_fair_share * job.slurm_fair + size_weight \
            + self.priority_weight_partition * job.slurm_partition + self.priority_weight_qos * job.slurm_qos \
            + self.tres_weight_cpu * job.slurm_tres_cpu

        return priority_score

    def fcfs_priority(self, job):
        return job.submit_time

    def smallest_job_first(self, job):
        return job.number_of_allocated_processors

    def largest_job_first(self, job):
        return sys.maxsize - job.number_of_allocated_processors

    def shortest_job_first(self, job):
        return job.request_time

    def longest_job_first(self, job):
        return sys.maxsize - job.request_time

    def _configure_slurm(self, age, fair_share, job_size, partition, qos, tres_cpu, max_age, favor_small):
        self.priority_weight_age = age
        self.priority_max_age = max_age
        self.priority_weight_fair_share = fair_share
        self.priority_weight_job_size = job_size
        self.priority_favor_small = favor_small
        self.priority_weight_partition = partition
        self.priority_weight_qos = qos
        self.tres_weight_cpu = tres_cpu

    def step_job_index(self, action):

        get_this_job_scheduled = False
        job_for_scheduling = self.job_queue[action]
        assert job_for_scheduling.job_id > 0

        if self.cluster.can_allocated(job_for_scheduling):
            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            if DEBUG:
                print("In step, schedule a job, ", job_for_scheduling, " with free nodes: ", self.cluster.free_node)
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                          job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            self.schedule_logs.append(job_for_scheduling)
            get_this_job_scheduled = True
            self.Metrics_Queue_Length += self._job_queue_size()
            self.Metrics_Probe_Times += 1
            self.Metrics_Total_Execution_Time = max(self.Metrics_Total_Execution_Time,
                                                    job_for_scheduling.scheduled_time +
                                                    job_for_scheduling.run_time)
            self.Metrics_Average_Slow_Down += (job_for_scheduling.scheduled_time - job_for_scheduling.submit_time)
            self.Metrics_Average_Response_Time += (job_for_scheduling.scheduled_time -
                                                   job_for_scheduling.submit_time + job_for_scheduling.run_time)
            self.Metrics_System_Utilization += (job_for_scheduling.run_time *
                                                job_for_scheduling.request_number_of_processors)
            self.job_queue[action] = Job()  # remove the job from job queue
        else:
            # if there is no enough resource for current job, try to backfill the jobs behind it
            # calculate the expected starting time of job[i].
            _needed_processors = job_for_scheduling.request_number_of_processors
            if DEBUG:
                print("try to back fill job, ", job_for_scheduling)
            _expected_start_time = self.current_timestamp
            _extra_released_processors = 0

            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            _released_resources = self.cluster.free_node * self.cluster.num_procs_per_node
            if DEBUG:
                print("total resources: ", self.cluster.total_node * self.cluster.num_procs_per_node)
                print("_released_resources, ", _released_resources)
            for _job in self.running_jobs:
                if DEBUG:
                    print("running job: ", _job)
                _released_resources += len(_job.allocated_machines) * self.cluster.num_procs_per_node
                released_time = _job.scheduled_time + _job.run_time
                if _released_resources >= _needed_processors:
                    _expected_start_time = released_time
                    _extra_released_processors = _released_resources - _needed_processors
                    break
            if DEBUG:
                print("_released_resources2, ", _released_resources)
            assert _released_resources >= _needed_processors

            # find do we have later jobs that do not affect the _expected_start_time
            for j in range(0, MAX_QUEUE_SIZE):
                _schedule_it = False
                _job = self.job_queue[j]
                if _job.job_id == 0:
                    continue
                assert _job.scheduled_time == -1  # this job should never be scheduled before.
                if not self.cluster.can_allocated(_job):
                    # if this job can not be allocated, skip to next job in the queue
                    continue

                if (_job.run_time + self.current_timestamp) <= _expected_start_time:
                    # if i can finish earlier than the expected start time of job[i], schedule it.
                    _schedule_it = True
                else:
                    if _job.request_number_of_processors < _extra_released_processors:
                        # if my allocation is small enough, not affecting anything, schedule it
                        _schedule_it = True
                        # but, we have to update the _extra_released_processors if we took it
                        _extra_released_processors -= _job.request_number_of_processors

                if _schedule_it:
                    if DEBUG:
                        print("take backfilling: ", _job, " for job, ", job_for_scheduling)
                    _job.scheduled_time = self.current_timestamp
                    _job.allocated_machines = self.cluster.allocate(_job.job_id, _job.request_number_of_processors)
                    self.running_jobs.append(_job)
                    self.schedule_logs.append(_job)
                    self.Metrics_Queue_Length += self._job_queue_size()
                    self.Metrics_Probe_Times += 1
                    self.Metrics_Total_Execution_Time = max(self.Metrics_Total_Execution_Time,
                                                            _job.scheduled_time + _job.run_time)
                    self.Metrics_Average_Slow_Down += (_job.scheduled_time - _job.submit_time)
                    self.Metrics_Average_Response_Time += (_job.scheduled_time - _job.submit_time + _job.run_time)
                    self.Metrics_System_Utilization += (_job.run_time * _job.request_number_of_processors)
                    self.job_queue[j] = Job()

        # move time forward
        if DEBUG:
            print("schedule job?", job_for_scheduling, "move time forward, get_this_job_scheduled: ",
                  get_this_job_scheduled,
                  " job queue is empty?: ", self._is_job_queue_empty(),
                  " running jobs?: ", len(self.running_jobs))

        while not get_this_job_scheduled or self._is_job_queue_empty():
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time \
                    and not self._is_job_queue_full():

                for i in range(0, MAX_QUEUE_SIZE):
                    if self.job_queue[i].job_id == 0:
                        self.job_queue[i] = self.loads[self.next_arriving_job_idx]
                        # current timestamp may be larger than next_arriving_job's submit time because job queue was
                        # full and we move forward to release resources.
                        self.current_timestamp = max(self.current_timestamp,
                                                     self.loads[self.next_arriving_job_idx].submit_time)
                        self.next_arriving_job_idx += 1
                        break
            else:
                if not self.running_jobs:
                    break
                self.current_timestamp = next_resource_release_time
                self.cluster.release(next_resource_release_machines)
                removed_job = self.running_jobs.pop(0)  # remove the first running job.
                if DEBUG:
                    print("In step, release a job, ", removed_job, " generated free nodes: ", self.cluster.free_node)

        if DEBUG:
            running_queue_machine_number = 0
            for _job in self.running_jobs:
                running_queue_machine_number += len(_job.allocated_machines) * self.cluster.num_procs_per_node
            total = running_queue_machine_number + \
                    self.cluster.free_node * self.cluster.num_procs_per_node

            print("Running jobs take ", running_queue_machine_number,
                  " Remaining free processors ", self.cluster.free_node * self.cluster.num_procs_per_node,
                  " total ", total)

        # calculate reward
        reward = 0.0

        done = True
        for i in range(self.start, self.last_job_in_batch):
            if self.loads[i].scheduled_time == -1:  # have at least one job in the batch who has not been scheduled
                done = False
                break

        obs = self.build_observation()

        # @update: we do not give reward until we finish scheduling everything.
        if done:
            execution_time = self.Metrics_Total_Execution_Time
            slow_down = self.Metrics_Average_Slow_Down
            response_time = self.Metrics_Average_Response_Time
            utilization = float(self.Metrics_System_Utilization) / float(self.cluster.num_procs_per_node *
                                                                         self.cluster.total_node *
                                                                         self.Metrics_Total_Execution_Time)
            average_queue_size = float(self.Metrics_Queue_Length) / float(self.Metrics_Probe_Times)

            if DEBUG:
                print("algorithm  *  total time: ", self.Metrics_Total_Execution_Time, " slow down: ",
                      self.Metrics_Average_Slow_Down, " response time: ", self.Metrics_Average_Response_Time,
                      " utility: ", utilization,
                      "average queue size:", average_queue_size)

            min_total = sys.maxsize
            min_slowdown = sys.maxsize
            min_response = sys.maxsize
            max_utilization = 0


            for i in range(0, 5):
                [total_ts, slow_ts, resp_ts, util_ts, _] = self.pre_processed_job_metrics[str(self.start)][str(i)]
                if DEBUG:
                    print("algorithm ", i, " total time: ", total_ts, " slow down: ",
                          slow_ts, " response time: ", resp_ts,
                          " utility: ", util_ts)
                if total_ts < min_total:
                    min_total = total_ts
                if slow_ts < min_slowdown:
                    min_slowdown = slow_ts
                if resp_ts < min_response:
                    min_response = resp_ts
                if util_ts > max_utilization:
                    max_utilization = util_ts

            if DEBUG:
                print("SlowDown. RL Agent:", slow_down, "Best of all:", min_slowdown)

            #if execution_time < min_total:
            #    reward += 1
            #else:
            #    reward -= 1
            reward += float(min_slowdown + 1) / float(slow_down + 1)
            #if utilization > max_utilization:
            #    reward += 1
            #else:
            #    reward -= 1

        return [obs, reward, done, None]

    def step(self, action):
        # action is one of the defined scheduler. 
        self.priority_function = self.scheduler_algs.get(action, self.fcfs_priority)
        all_jobs = list(self.job_queue)
        all_jobs.sort(key=lambda j: (self.priority_function(j)))
        # self.job_queue.sort(key=lambda j: (self.priority_function(j)))

        top_job_to_pick = all_jobs[0]
        for j in all_jobs:
            if j.job_id != 0:
                top_job_to_pick = j
        assert top_job_to_pick.job_id != 0

        job_for_scheduling = None
        job_for_scheduling_index = -1
        for idx in range(0, MAX_QUEUE_SIZE):
            if self.job_queue[idx].job_id == top_job_to_pick.job_id:
                job_for_scheduling = self.job_queue[idx]
                job_for_scheduling_index = idx
                break
        assert job_for_scheduling is not None
        assert job_for_scheduling_index != -1

        get_this_job_scheduled = False

        if self.cluster.can_allocated(job_for_scheduling):
            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            if DEBUG:
                print("In step, schedule a job, ", job_for_scheduling, " with free nodes: ", self.cluster.free_node)
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                          job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            self.schedule_logs.append(job_for_scheduling)
            get_this_job_scheduled = True
            self.Metrics_Queue_Length += self._job_queue_size()
            self.Metrics_Probe_Times += 1
            self.Metrics_Total_Execution_Time = max(self.Metrics_Total_Execution_Time,
                                                    job_for_scheduling.scheduled_time +
                                                    job_for_scheduling.run_time)
            self.Metrics_Average_Slow_Down += (job_for_scheduling.scheduled_time - job_for_scheduling.submit_time)
            self.Metrics_Average_Response_Time += (job_for_scheduling.scheduled_time -
                                                   job_for_scheduling.submit_time + job_for_scheduling.run_time)
            self.Metrics_System_Utilization += (job_for_scheduling.run_time *
                                                job_for_scheduling.request_number_of_processors)
            self.job_queue[job_for_scheduling_index] = Job()  # remove the job from job queue
        else:
            # if there is no enough resource for current job, try to backfill the jobs behind it
            # calculate the expected starting time of job[i].
            _needed_processors = job_for_scheduling.request_number_of_processors
            if DEBUG:
                print("try to back fill job, ", job_for_scheduling)
            _expected_start_time = self.current_timestamp
            _extra_released_processors = 0

            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            _released_resources = self.cluster.free_node * self.cluster.num_procs_per_node
            if DEBUG:
                print("total resources: ", self.cluster.total_node * self.cluster.num_procs_per_node)
                print("_released_resources, ", _released_resources)
            for _job in self.running_jobs:
                if DEBUG:
                    print("running job: ", _job)
                _released_resources += len(_job.allocated_machines) * self.cluster.num_procs_per_node
                released_time = _job.scheduled_time + _job.run_time
                if _released_resources >= _needed_processors:
                    _expected_start_time = released_time
                    _extra_released_processors = _released_resources - _needed_processors
                    break
            if DEBUG:
                print("_released_resources2, ", _released_resources)
            assert _released_resources >= _needed_processors

            # find do we have later jobs that do not affect the _expected_start_time
            for j in range(0, MAX_QUEUE_SIZE):
                _schedule_it = False
                _job = self.job_queue[j]
                if _job.job_id == 0:
                    continue
                assert _job.scheduled_time == -1  # this job should never be scheduled before.
                if not self.cluster.can_allocated(_job):
                    # if this job can not be allocated, skip to next job in the queue
                    continue

                if (_job.run_time + self.current_timestamp) <= _expected_start_time:
                    # if i can finish earlier than the expected start time of job[i], schedule it.
                    _schedule_it = True
                else:
                    if _job.request_number_of_processors < _extra_released_processors:
                        # if my allocation is small enough, not affecting anything, schedule it
                        _schedule_it = True
                        # but, we have to update the _extra_released_processors if we took it
                        _extra_released_processors -= _job.request_number_of_processors

                if _schedule_it:
                    if DEBUG:
                        print("take backfilling: ", _job, " for job, ", job_for_scheduling)
                    _job.scheduled_time = self.current_timestamp
                    _job.allocated_machines = self.cluster.allocate(_job.job_id, _job.request_number_of_processors)
                    self.running_jobs.append(_job)
                    self.schedule_logs.append(_job)
                    self.Metrics_Queue_Length += self._job_queue_size()
                    self.Metrics_Probe_Times += 1
                    self.Metrics_Total_Execution_Time = max(self.Metrics_Total_Execution_Time,
                                                            _job.scheduled_time + _job.run_time)
                    self.Metrics_Average_Slow_Down += (_job.scheduled_time - _job.submit_time)
                    self.Metrics_Average_Response_Time += (_job.scheduled_time - _job.submit_time + _job.run_time)
                    self.Metrics_System_Utilization += (_job.run_time * _job.request_number_of_processors)
                    self.job_queue[j] = Job()

        # move time forward
        if DEBUG:
            print("schedule job?", job_for_scheduling, "move time forward, get_this_job_scheduled: ",
                  get_this_job_scheduled,
                  " job queue is empty?: ", self._is_job_queue_empty(),
                  " running jobs?: ", len(self.running_jobs))

        while not get_this_job_scheduled or self._is_job_queue_empty():
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time \
                    and not self._is_job_queue_full():

                for i in range(0, MAX_QUEUE_SIZE):
                    if self.job_queue[i].job_id == 0:
                        self.job_queue[i] = self.loads[self.next_arriving_job_idx]
                        # current timestamp may be larger than next_arriving_job's submit time because job queue was
                        # full and we move forward to release resources.
                        self.current_timestamp = max(self.current_timestamp,
                                                     self.loads[self.next_arriving_job_idx].submit_time)
                        self.next_arriving_job_idx += 1
                        break
            else:
                if not self.running_jobs:
                    break
                self.current_timestamp = next_resource_release_time
                self.cluster.release(next_resource_release_machines)
                removed_job = self.running_jobs.pop(0)  # remove the first running job.
                if DEBUG:
                    print("In step, release a job, ", removed_job, " generated free nodes: ", self.cluster.free_node)

        if DEBUG:
            running_queue_machine_number = 0
            for _job in self.running_jobs:
                running_queue_machine_number += len(_job.allocated_machines) * self.cluster.num_procs_per_node
            total = running_queue_machine_number + \
                    self.cluster.free_node * self.cluster.num_procs_per_node

            print("Running jobs take ", running_queue_machine_number,
                  " Remaining free processors ", self.cluster.free_node * self.cluster.num_procs_per_node,
                  " total ", total)

        # calculate reward
        reward = 0.0

        done = True
        for i in range(self.start, self.last_job_in_batch):
            if self.loads[i].scheduled_time == -1:  # have at least one job in the batch who has not been scheduled
                done = False
                break

        obs = self.build_observation()

        # @update: we do not give reward until we finish scheduling everything.
        if done:
            execution_time = self.Metrics_Total_Execution_Time
            slow_down = self.Metrics_Average_Slow_Down
            response_time = self.Metrics_Average_Response_Time
            utilization = float(self.Metrics_System_Utilization) / float(self.cluster.num_procs_per_node *
                                                                         self.cluster.total_node *
                                                                         self.Metrics_Total_Execution_Time)
            average_queue_size = float(self.Metrics_Queue_Length) / float(self.Metrics_Probe_Times)

            if DEBUG:
                print("algorithm  *  total time: ", self.Metrics_Total_Execution_Time, " slow down: ",
                      self.Metrics_Average_Slow_Down, " response time: ", self.Metrics_Average_Response_Time,
                      " utility: ", utilization,
                      "average queue size:", average_queue_size)

            min_total = sys.maxsize
            min_slowdown = sys.maxsize
            min_response = sys.maxsize
            max_utilization = 0

            for i in range(0, 5):
                [total_ts, slow_ts, resp_ts, util_ts, _] = self.pre_processed_job_metrics[str(self.start)][str(i)]
                if DEBUG:
                    print("algorithm ", i, " total time: ", total_ts, " slow down: ",
                          slow_ts, " response time: ", resp_ts,
                          " utility: ", util_ts)
                if total_ts < min_total:
                    min_total = total_ts
                if slow_ts < min_slowdown:
                    min_slowdown = slow_ts
                if resp_ts < min_response:
                    min_response = resp_ts
                if util_ts > max_utilization:
                    max_utilization = util_ts

            if DEBUG:
                print("SlowDown. RL Agent:", slow_down, "Best of all:", min_slowdown)

            # if execution_time < min_total:
            #    reward += 1
            # else:
            #    reward -= 1
            reward += float(min_slowdown + 1) / float(slow_down + 1)
            # if utilization > max_utilization:
            #    reward += 1
            # else:
            #    reward -= 1

        return [obs, reward, done, None]


def legal(obs, action):
    q = np.reshape(obs, [8, 8, 3])
    if all(q[int(action / 8), int(action % 8)] == 0) or action == 63:
        return False
    return True

def heuristic(env, s):
    # [print(_, end=", ") for _ in s[:-1]]
    # print("")
    #actions = np.argsort(np.random.dirichlet(np.ones(64), size=1)).ravel()
    #legal_actions = [a for a in actions if legal(s, a)]
    #assert legal_actions
    #return legal_actions[0]
    return np.random.randint(0, 5)

def demo_scheduler(env):
    env.seed()
    total_reward = 0.0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, _ = env.step(a)
        total_reward += r

        if done:
            #print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.4f}".format(steps, total_reward))

        steps += 1
        if done:
            break
    return total_reward


if __name__ == '__main__':
    print(os.getcwd())
    env = HpcEnv()
    env.my_init(workload_file = '../../../data/RICC-2010-2.swf', pre_processed_metrics_file="../../../data/RICC-RL-200.txt")
    '''
    for i in range(0, 5):
        ts, _ = env.get_metrics_using_algorithm(i, 12897, 13532)
        print("algorithm ", i, " execute: ", ts)
    '''
    for i in range(0, 100):
        demo_scheduler(env)