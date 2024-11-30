import numpy as np
import tensorflow as tf
import gym
import os
import sys
import time
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import mpi_avg
from spinup.utils.logx import restore_tf_graph
import os.path as osp
from HPCSimPickJobs import *

class SACBuffer:
    """
    A buffer for storing trajectories experienced by an SAC agent interacting
    with the environment. Stores state-action pairs, rewards, next states, and done flags.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        size = size
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(size, MAX_QUEUE_SIZE), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)

        self.next_obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        # Q-values for both Q1 and Q2
        self.q1_buf = np.zeros(size, dtype=np.float32)
        self.q2_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, mask, rew, next_obs, done, q1, q2, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.q1_buf[self.ptr] = q1
        self.q2_buf[self.ptr] = q2
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self):
        """
        This method is no longer needed in SAC, since SAC does not compute advantages using GAE-Lambda.
        However, you can still use it to normalize the rewards or other steps if needed.
        """
        pass

    def get(self):
        assert self.ptr < self.max_size  # Check if buffer has been fully populated
        actual_size = self.ptr
        self.ptr = 0  # Reset the pointer for next trajectory

        # Return the data for SAC update
        return [
            self.obs_buf[:actual_size],        # Observations
            self.act_buf[:actual_size],        # Actions
            self.rew_buf[:actual_size],        # Rewards
            self.next_obs_buf[:actual_size],   # Next observations
            self.done_buf[:actual_size],       # Done flags
            self.q1_buf[:actual_size],         # Q1 values
            self.q2_buf[:actual_size]          # Q2 values
        ]

def attention(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE, JOB_FEATURES])
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

def rl_kernel(x, act_dim):
    x = tf.reshape(x, shape=[-1,MAX_QUEUE_SIZE, JOB_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    return x

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


def stochastic_policy(x, a_ph, mask, action_space, attn=False):
    """
    Generates a stochastic policy based on the input state.
    Samples actions from the policy distribution.
    """
    act_dim = action_space.n  # Number of discrete actions
    
    if attn:
        # If attention is enabled, use an attention mechanism to compute the policy output
        output_layer = attention(x, act_dim)
    else:
        # Otherwise, use a regular fully connected network
        output_layer = rl_kernel(x, act_dim)
    
    # Apply mask to actions (for invalid actions)
    output_layer = output_layer + (mask - 1) * 1000000 
    
    # Compute log-probabilities for each action
    logp_all = tf.nn.log_softmax(output_layer)
    
    
    pi = tf.squeeze(tf.multinomial(output_layer, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a_ph, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    
    return pi, logp, logp_pi, output_layer

def actor_critic(x, a, mask, action_space=None, attn=False):
    """
    Builds the policy and value networks for SAC.
    - Policy Network: Outputs actions and log probabilities.
    - Q1 and Q2 Networks: Estimate action values.
    - Value Network: Stabilizes training.
    """
    with tf.variable_scope('pi'):
        # Policy network outputs sampled actions and their log probabilities
        pi, logp, logp_pi, out = stochastic_policy(x, a, mask, action_space, attn)
    with tf.variable_scope('q1'):
        # First Q-value network
        q1 = tf.squeeze(critic_mlp(tf.concat([x, a], axis=-1), 1), axis=1)
    with tf.variable_scope('q2'):
        # Second Q-value network
        q2 = tf.squeeze(critic_mlp(tf.concat([x, a], axis=-1), 1), axis=1)
    with tf.variable_scope('v'):
        # Value network
        v = tf.squeeze(critic_mlp(x, 1), axis=1)
        
    return pi, logp, logp_pi, v, out, q1, q2

def sac(workload_file, model_path, ac_kwargs=dict(), gamma=0.99, polyak=0.995, alpha=0.2,
        lr=3e-4, epochs=50, batch_size=256, steps_per_epoch=4000, 
        start_steps=10000, update_after=1000, update_every=50,
        logger_kwargs=dict(), seed=0, max_ep_len=1000, shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0,
        attn=False, train_pi_iters=80, target_kl=0.01):
    
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Initialize environment
    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type, batch_job_slice=batch_job_slice, build_sjf=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn

    # Replay Buffer
    replay_buffer =  SACBuffer(obs_dim, act_dim, steps_per_epoch * JOB_SEQUENCE_SIZE)

    # Placeholders
    x_ph, a_ph = placeholders_from_spaces(env.observation_space, env.action_space)
    mask_ph = placeholder(MAX_QUEUE_SIZE)
    r_ph, x_next_ph, d_ph = placeholders(None, env.observation_space.shape, None)

    # Build computation graph
    pi, logp, logp_pi, v, out, q1, q2= actor_critic(x_ph, a_ph, mask_ph, **ac_kwargs)

    all_phs = [x_ph, a_ph, mask_ph, r_ph, x_next_ph, d_ph]

    # Get action operations for SAC
    get_action_ops = [pi, v, logp_pi, q1, q2]

    # Count variables for logging (pi, v, q1, q2)
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v', 'q1', 'q2'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t q1: %d, \t q2: %d\n' % var_counts)


    # # Target value computation
    target_v = tf.squeeze(critic_mlp(x_next_ph, 1), axis=1)
    q_backup = r_ph + gamma * (1 - d_ph) * target_v

    # Loss functions
    q1_loss = tf.reduce_mean((q1 - q_backup) ** 2)
    q2_loss = tf.reduce_mean((q2 - q_backup) ** 2)

    # Value loss for SAC
    v_backup = tf.minimum(q1, q2) - alpha * logp_pi
    v_loss = tf.reduce_mean((v - tf.stop_gradient(v_backup)) ** 2)

    # Policy iteration with entropy regularization
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1)

    # Optimizers
    q1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    q2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Training steps
    train_q1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(q1_loss)
    train_q2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(q2_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=lr).minimize(v_loss)
    train_pi = tf.train.AdamOptimizer(learning_rate=lr).minimize(pi_loss)

    # Info, useful to watch during learning

    approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.add_to_collection("train_pi", train_pi)
    tf.add_to_collection("train_v", train_v)
    tf.add_to_collection("train_q1", train_q1)
    tf.add_to_collection("train_q2", train_q2)

    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph, 'mask': mask_ph, 'r': r_ph, 'x_next': x_next_ph, 'done': d_ph},
                          outputs={'pi': pi, 'v': v, 'q1_loss': q1_loss, 'q2_loss': q2_loss, 'v_loss': v_loss, 'policy_loss': policy_loss})


    # SAC Update Functions
    @tf.function
    def update():

        inputs = {k:v for k,v in zip(all_phs, replay_buffer.get())}
        pi_l_old, v_l_old, q1_l_old, q2_l_old, ent = sess.run([pi_loss, v_loss, q1_loss, q2_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)

        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, q1_l_new, q2_l_new, kl, cf = sess.run([pi_loss, v_loss, q1_loss, q2_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     LossQ1=q1_l_old, LossQ2=q2_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))
        
    # Main Loop: Collect experince in env and update/log each epoch
    start_time = time.time()
    [o, co], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0,0,0,0

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    num_total = 0

    for epoch in range(epochs):
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

            a, v_t, logp_t, output = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1), mask_ph: np.array(lst).reshape(1,-1)})
            # print(a, end=" ")

            num_total += 1
            
        print("hi")





        # Sample a batch from the replay buffer
        batch = replay_buffer.sample_batch(batch_size)

        # Unpack batch
        obs, act, rew, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

        # Target value computation for SAC
        target_v = tf.squeeze(target_value_net(next_obs), axis=-1)
        target_q = rew + gamma * (1 - done) * target_v

        # Q-network losses
        with tf.GradientTape() as q1_tape, tf.GradientTape() as q2_tape:
            q1 = tf.squeeze(q1_net(obs), axis=-1)
            q2 = tf.squeeze(q2_net(obs), axis=-1)
            q1_loss = tf.reduce_mean((q1 - target_q) ** 2)
            q2_loss = tf.reduce_mean((q2 - target_q) ** 2)
        q1_grads = q1_tape.gradient(q1_loss, q1_net.trainable_variables)
        q2_grads = q2_tape.gradient(q2_loss, q2_net.trainable_variables)
        q1_optimizer.apply_gradients(zip(q1_grads, q1_net.trainable_variables))
        q2_optimizer.apply_gradients(zip(q2_grads, q2_net.trainable_variables))

        # Value network loss
        with tf.GradientTape() as value_tape:
            min_q = tf.minimum(q1, q2)
            v = tf.squeeze(value_net(obs), axis=-1)
            value_loss = tf.reduce_mean((v - (min_q - alpha * tf.math.log(policy_net(obs)))) ** 2)
        value_grads = value_tape.gradient(value_loss, value_net.trainable_variables)
        value_optimizer.apply_gradients(zip(value_grads, value_net.trainable_variables))

        # Policy network loss
        with tf.GradientTape() as policy_tape:
            log_pi = tf.math.log(policy_net(obs))
            policy_loss = tf.reduce_mean(alpha * log_pi - q1)
        policy_grads = policy_tape.gradient(policy_loss, policy_net.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_grads, policy_net.trainable_variables))

        # Update target value network (polyak averaging)
        for v, target_v in zip(value_net.trainable_variables, target_value_net.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)

    # Main Training Loop
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            # Random actions for exploration
            if t > start_steps:
                a = policy_net(o).numpy()
            else:
                a = env.action_space.sample()

            # Step in environment
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Store in replay buffer
            replay_buffer.store(o, a, r, next_o, d)

            # Update observation
            o = next_o

            if d or ep_len == max_ep_len:
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Update networks
            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    update()

        # Logging
        logger.log_tabular('Epoch', epoch)
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
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./data/logs/ppo_temp/ppo_temp_s0')
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

    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=0)
    sac(args.workload, args.model, gamma=args.gamma, epochs=args.epochs, logger_kwargs=logger_kwargs)