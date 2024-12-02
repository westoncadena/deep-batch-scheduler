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
import torch

def rl_kernel(x, act_dim):
    x = tf.reshape(x, shape=[-1,MAX_QUEUE_SIZE, JOB_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    return x

def attention(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE, JOB_FEATURES])
    # x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    q = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    k = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    v = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    score = tf.matmul(q,tf.transpose(k,[0,2,1]))
    score = tf.nn.softmax(score,-1)
    attn = tf.reshape(score,(-1, MAX_QUEUE_SIZE, MAX_QUEUE_SIZE))
    x = tf.matmul(attn, v)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)

    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    return x

"""
Policies
"""
def categorical_policy(x, a, mask, action_space, attn):
    act_dim = action_space.n
    if attn:
        output_layer = attention(x, act_dim)
    else:
        output_layer = rl_kernel(x, act_dim)
    output_layer = output_layer+(mask-1)*1000000
    logp_all = tf.nn.log_softmax(output_layer)

    pi = tf.squeeze(tf.multinomial(output_layer, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, output_layer

"""
Actor-Critics
"""
def critic_mlp(x, act_dim):
    x = tf.reshape(x, shape=[-1,MAX_QUEUE_SIZE, JOB_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)

    return tf.layers.dense(x, units=act_dim)


def actor_critic(x, a, mask, act_dim=None, action_space=None, attn=False):
    with tf.variable_scope('pi'):
        pi, logp, logp_pi , out= categorical_policy(x, a, mask, action_space, attn)  # Policy
    with tf.variable_scope('q1'):
        q1 = critic_mlp(x, action_space.n)  # Q-value for action a
    with tf.variable_scope('q2'):
        q2 = critic_mlp(x, action_space.n)  # Second Q-value for stability
    with tf.variable_scope('v'):
        v = tf.squeeze(critic_mlp(x, 1), axis=1)  # Value function
    return pi, logp, logp_pi, q1, q2, v, out

class SACBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        size = size * 5  # assume the traj can be really long
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(size, MAX_QUEUE_SIZE), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.gamma, self.lam = gamma, lam

    def store(self, obs, act, next_obs, mask, rew, done, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def get(self):
        """
        Get the data from the buffer with normalized advantages.
        """
        assert self.ptr <= self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0

        # actual_adv_buf = np.array(self.adv_buf, dtype=np.float32)[:actual_size]
        # adv_mean = np.mean(actual_adv_buf)
        # adv_std = np.std(actual_adv_buf)
        # actual_adv_buf = (actual_adv_buf - adv_mean) / (adv_std + 1e-8)

        return [
            self.obs_buf[:actual_size], self.act_buf[:actual_size], self.mask_buf[:actual_size],
            self.rew_buf[:actual_size], self.next_obs_buf[:actual_size], self.done_buf[:actual_size]
        ]


def sac(workload_file, model_path, ac_kwargs=dict(), seed=0, 
        traj_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_q_iters=80, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,pre_trained=0,trained_model=None,attn=False,shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0, alpha = 0.2):
    
    # Initializeing Logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Initializing Random Seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Initialize Enviornment
    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type, batch_job_slice=batch_job_slice, build_sjf=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)
    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn

    # Inputs to computation graph
    buf = SACBuffer(obs_dim, act_dim, traj_per_epoch * JOB_SEQUENCE_SIZE, gamma, lam)

    if pre_trained:
        print("pretrained")
    else:
        x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)

        mask_ph = placeholder(MAX_QUEUE_SIZE)
        r_ph, done_ph, next_obs_ph= placeholders(None, None, env.observation_space.shape)

        # Main outputs from computation graph
        pi, logp, logp_pi, q1, q2, v, out = actor_critic(x_ph, a_ph, mask_ph, **ac_kwargs)

        # All Placeholders in the order (used to zip from buffer)
        all_phs = [x_ph, a_ph, mask_ph, r_ph, next_obs_ph, done_ph]


        get_action_ops = [pi, q1, q2, v, logp_pi, out]  

        # Count variables for policy, two Q-value networks, value network, and alpha (entropy coefficient)
        var_counts = tuple(count_vars(scope) for scope in ['pi', 'q1', 'q2', 'v'])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t v: %d \n' % var_counts)

        # SAC Objectives
        a_ph_expanded = tf.expand_dims(a_ph, axis=1)
        q1_selected = tf.gather(q1, a_ph_expanded, axis=1, batch_dims=1)
        q2_selected = tf.gather(q2, a_ph_expanded, axis=1, batch_dims=1)
        v_target = tf.minimum(q1_selected, q2_selected) - alpha * logp_pi
        q1_target = r_ph + gamma * (1 - done_ph) * (v_target - alpha * logp_pi)
        q2_target = r_ph + gamma * (1 - done_ph) * (v_target - alpha * logp_pi)
        q1_loss = 0.5 * tf.reduce_mean((q1_selected - q1_target) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((q2_selected - q2_target) ** 2)

        pi_loss = tf.reduce_mean(alpha * logp_pi - q1_selected)

        v_loss = 0.5 * tf.reduce_mean((v - v_target) ** 2)

        # Info (useful to watch during learning)
        approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute

        # Optimizers
        train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
        train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
        train_q1 = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(q1_loss)
        train_q2 = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(q2_loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.add_to_collection("train_pi", train_pi)
        tf.add_to_collection("train_v", train_v)
        tf.add_to_collection("train_q1", train_q1)
        tf.add_to_collection("train_q2", train_q2)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a':a_ph, 'mask':mask_ph, 'r_ph':r_ph, 'done_ph':done_ph},
                          outputs={'pi': pi, 'v': v, 'q1':q1, 'q2':q2, 'out':out, 'pi_loss':pi_loss, 'logp': logp, 'logp_pi':logp_pi, 'v_loss':v_loss, 'approx_ent':approx_ent})


    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}

        a, q1_t, q2_t, v_t, logp_t, output = sess.run(get_action_ops, feed_dict=inputs)


        pi_l_old, v_l_old, q1_l_old, q2_l_old, ent = sess.run([pi_loss, v_loss, q1_loss, q2_loss, approx_ent], feed_dict=inputs)

        # Training Q-function and value function
        for _ in range(train_q_iters):
            # Update the Q-values and value function
            sess.run([train_q1, train_q2], feed_dict=inputs)

        for _ in range(train_v_iters):
            # Update the value function (v_loss)
            sess.run(train_v, feed_dict=inputs)

        # Training Policy
        for _ in range(train_pi_iters):
            # Update the policy by minimizing the policy loss
            sess.run(train_pi, feed_dict=inputs)

        q1_l_new, q2_l_new, v_l_new, pi_l_new = sess.run(
        [q1_loss, q2_loss, v_loss, pi_loss], feed_dict=inputs)

        logger.store(
            LossQ1=q1_l_old, LossQ2=q2_l_old, LossV=v_l_old, LossPi=pi_l_old,
            Entropy=ent,
            DeltaLossQ1=(q1_l_new - q1_l_old),
            DeltaLossQ2=(q2_l_new - q2_l_old),
            DeltaLossV=(v_l_new - v_l_old),
            DeltaLossPi=(pi_l_new - pi_l_old),
        )


    start_time = time.time()
    [o, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0,0,0,0

    num_total = 0
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        t = 0
        
        while True:

            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i:i+JOB_FEATURES] == [0]+[1]*(JOB_FEATURES-2)+[0]):
                    lst.append(0)
                elif all(o[i:i+JOB_FEATURES] == [1]*JOB_FEATURES):
                    lst.append(0)
                else:
                    lst.append(1)
            
            a, q1_t, q2_t, v_t, logp_t, output = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1), mask_ph: np.array(lst).reshape(1,-1)})

            num_total += 1

            o_old = o
            o, r, d, r2, sjf_t, f1_t = env.step(a[0])

            # save and log
            buf.store(o_old, a, o, np.array(lst), r, d, v_t)
            logger.store(VVals=v_t)

            # # save and log
            # buf.store(o, a, next_o, np.array(lst), r)
            # logger.store(VVals=v_t)

            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t

            if d:
                t += 1
                
                # print("Traj")
                # buf.finish_path(r)
                logger.store(EpRet=ep_ret, EpLen=ep_len, ShowRet=show_ret, SJF=sjf, F1=f1)
                [o, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0, 0, 0, 0
                if t >= traj_per_epoch:
                    
                    # print ("state:", state, "\nlast action in a traj: action_probs:\n", action_probs, "\naction:", action)
                    break

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)
        print("Updating")
        update()

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)* traj_per_epoch * JOB_SEQUENCE_SIZE)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('ShowRet', average_only=True)
        logger.log_tabular('SJF', average_only=True)
        logger.log_tabular('F1', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--model', type=str, default='./data/lublin_256.schd')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./data/logs/sac/sac_temp_s0')
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    
    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './data/logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)
    if args.pre_trained:
        model_file = os.path.join(current_dir, args.trained_model)
        # get_probs, get_value = load_policy(model_file, 'last')

        sac(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained=1,trained_model=os.path.join(model_file,"simple_save"),attn=args.attn,
            shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, score_type=args.score_type,
            batch_job_slice=args.batch_job_slice)
    else:
        sac(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained=0, attn=args.attn,shuffle=args.shuffle, backfil=args.backfil,
            skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice)
