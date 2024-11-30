import numpy as np
import tensorflow as tf
import gym
import os
import sys
import time
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.logx import restore_tf_graph
import os.path as osp
from HPCSimPickJobs import *

class ReplayBuffer:
    """ Replay Buffer for DQN """
    def __init__(self, size, obs_dim, act_dim):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs]
        )
        

def dqn(workload_file, model_path, seed=0, epochs=50, steps_per_epoch=4000, gamma=0.99,
        lr=1e-3, batch_size=32, buffer_size=100000, epsilon_decay=0.995, min_epsilon=0.1,
        logger_kwargs=dict(), save_freq=10, backfill=False, skip=False, shuffle=False,
        score_type=0, batch_job_slice=0):
    
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = HPCEnv(shuffle=shuffle, backfill=backfill, skip=skip, job_score_type=score_type, batch_job_slice=batch_job_slice, build_sjf=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about the action space with the policy?
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn

    replay_buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)

    # Placeholders
    obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    next_obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
    act_ph = tf.placeholder(tf.int32, shape=(None,))
    rew_ph = tf.placeholder(tf.float32, shape=(None,))
    done_ph = tf.placeholder(tf.float32, shape=(None,))

    q_net = build_q_network(obs_ph, act_dim)
    target_q_net = build_q_network(obs_ph, act_dim)

    # Loss and Optimizer
    action_q_vals = tf.reduce_sum(q_net * tf.one_hot(act_ph, act_dim), axis=1)
    target_q_vals = rew_ph + gamma * (1 - done_ph) * tf.reduce_max(target_q_net, axis=1)
    loss = tf.reduce_mean((action_q_vals - tf.stop_gradient(target_q_vals)) ** 2)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a':a_ph, 'adv':adv_ph, 'mask':mask_ph, 'ret':ret_ph, 'logp_old_ph':logp_old_ph}, outputs={'pi': pi, 'v': v, 'out':out, 'pi_loss':pi_loss, 'logp': logp, 'logp_pi':logp_pi, 'v_loss':v_loss, 'approx_ent':approx_ent, 'approx_kl':approx_kl, 'clipped':clipped, 'clipfrac':clipfrac})

    # Main loop: collect experince in env and update/log each epoch
    start_time = time.time()
    epsilon = 0.1

    for epoch in range(epochs):
        # reset the enviorment
        [obs, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0,0,0,0

        for step in range(steps_per_epoch):
            # Epsilon-greedy action selection
            if np.random.ran() < epsilon:
                act = env.action_space.sample
            else:
                q_vals = sess.run(q_net, feed_dict={obs_ph: obs.reshape(1, -1)})
                act = np.argmax(q_vals)

            next_obs, rew, done, r2, sjf_t, f1_t = env.step(act[0])

            ep_ret += rew
            ep_len += 1

            # Store experince in replay buffer
            replay_buffer.store(obs, act, rew, next_obs, done)

            obs = next_obs

            # Training step
            if replay_buffer.size >= batch_size:
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {
                    obs_ph: batch['obs'],
                    next_obs_ph: batch['next_obs'],
                    act_ph: batch['act'],
                    rew_ph: batch['rew'],
                    done_ph: batch['done']
                }
                sess.run(train_op, feed_dict)

            if done or (step == steps_per_epoch - 1):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, ep_ret, ep_len = env.reset(), 0, 0

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Log information
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

        # Save model
        if epoch % save_freq == 0 or epoch == epochs - 1:
            logger.save_state({'env': env}, None)