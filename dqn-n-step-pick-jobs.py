import numpy as np
import tensorflow as tf
import gym
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.logx import restore_tf_graph
import os.path as osp
from HPCSimPickJobs import *

class DQNReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, gamma, n_step):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.masks = np.zeros((buffer_size, action_dim), dtype=np.float32)

        self.gamma = gamma
        self.n_step = n_step
        self.ptr, self.size, self.max_size = 0, 0, buffer_size

        # Temporary buffer for current trajectory
        self.traj_states = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_next_states = []
        self.traj_dones = []
        self.traj_masks = []

    def store(self, state, action, reward, next_state, done, mask):
        """
        Store a transition in the trajectory buffer.
        """
        self.traj_states.append(state)
        self.traj_actions.append(action)
        self.traj_rewards.append(reward)
        self.traj_next_states.append(next_state)
        self.traj_dones.append(done)
        self.traj_masks.append(mask)

        if done:
            self.finish_path()

    def finish_path(self):
        """
        Process the stored trajectory to compute n-step returns and add to the main buffer.
        """
        n = len(self.traj_rewards)
        
        for i in range(n):
            # Compute n-step return starting from the current step
            R = 0
            gamma_pow = 1
            self.n_step = n
            for j in range(i, min(i + self.n_step, n)):
                R += gamma_pow * self.traj_rewards[j]
                gamma_pow *= self.gamma
                if self.traj_dones[j]:  # Stop summing if trajectory ends
                    break

            next_state = self.traj_next_states[min(i + self.n_step - 1, n - 1)]
            done = self.traj_dones[min(i + self.n_step - 1, n - 1)]
            mask = self.traj_masks[min(i + self.n_step - 1, n - 1)]

            # Store n-step transition in the main buffer
            self._add_to_buffer(self.traj_states[i], self.traj_actions[i], R, next_state, done, mask)

        # Clear trajectory buffer after processing
        self.traj_states = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_next_states = []
        self.traj_dones = []
        self.traj_masks = []

    def _add_to_buffer(self, state, action, reward, next_state, done, mask):
        """
        Add processed transition to the main replay buffer.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.masks[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
            self.masks[idxs]
        )
    
class QFunction(nn.Module):
    """
    Q-network definition using a flexible hidden layer structure.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = obs
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)

class DQNAgentN:
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 128, 64], lr=1e-3, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Define Q-networks using QFunction
        self.q_net = QFunction(state_dim, action_dim, hidden_sizes)
        self.target_q_net = QFunction(state_dim, action_dim, hidden_sizes)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # Copy initial weights

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state, epsilon, mask):
        """
        Select an action using epsilon-greedy policy with the provided mask.
        The mask indicates which actions are valid (1) or invalid (0).
        """
        if np.random.random() < epsilon:
            # Random action selection, but only choose from valid actions
            valid_actions = np.where(mask == 1)[0]
            return np.random.choice(valid_actions)  # Randomly select among valid actions
        
        # Compute Q-values for each action
        state_tensor = torch.FloatTensor(state).unsqueeze(0) 
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        # Mask out invalid actions by setting their Q-values to a very low value
        q_values[0, mask_tensor == 0] = -float('inf')  

        # Select the action with the highest Q-value among valid actions
        return torch.argmax(q_values).item()

    def update(self, states, actions, rewards, next_states, dones, masks):
        """
        Update the Q-network using a batch of transitions.
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # Convert to (batch, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Convert to (batch, 1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)  # Convert to (batch, 1)

        # Compute Q(s, a) from the main network
        q_values = self.q_net(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            # max_next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            # target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
            next_q_values = self.target_q_net(next_states)
            mask_tensor = torch.tensor(masks, dtype=torch.bool)  # Convert mask to a boolean tensor
            next_q_values[mask_tensor == 0] = -float('inf')

            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]

            # Calculate target Q-values using the reward and the max Q-values from the next state
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)



        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update of the target network
        self.soft_update(self.q_net, self.target_q_net)

    def soft_update(self, online_net, target_net):
        """
        Soft update target network parameters using online network parameters.
        """
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)



def train_dqn(workload_file, model_path, ac_kwargs=dict(), seed=0, 
        traj_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,pre_trained=0,trained_model=None,attn=False,shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0):

        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        tf.set_random_seed(seed)
        np.random.seed(seed)

        # Initialize Enviornment
        env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type, batch_job_slice=batch_job_slice, build_sjf=False)
        env.seed(seed)
        env.my_init(workload_file=workload_file, sched_file=model_path)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        print("State dimension:", state_dim)
        print("Action dimension:", action_dim)

        # Initialize Agent
        agent = DQNAgentN(state_dim, action_dim, gamma=gamma)
        buffer = DQNReplayBuffer(state_dim, action_dim, buffer_size=10000, gamma=gamma, n_step=5)

        start_time = time.time()
        epsilon = 0.05
        
        for epoch in range(epochs):
            print(f"epoch {epoch}")

            [state, co], reward, done, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0,0,0,0
            trajectory = 0
            steps_per_traj = 0

            while True:
                steps_per_traj += 1
                
                mask = []
                for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                    if all(state[i:i+JOB_FEATURES] == [0]+[1]*(JOB_FEATURES-2)+[0]):
                        mask.append(0)
                    elif all(state[i:i+JOB_FEATURES] == [1]*JOB_FEATURES):
                        mask.append(0)
                    else:
                        mask.append(1)
                mask_arr = np.array(mask)

                action = agent.select_action(state, epsilon, mask_arr)


                next_state, reward, done, reward2, sjf_t, f1_t = env.step(action)
                ep_ret += reward
                ep_len += 1
                show_ret += reward2
                sjf += sjf_t
                f1 += f1_t

                buffer.store(state, action, reward, next_state, done, mask_arr)

                state = next_state

                if done:
                    logger.store(VVals=reward)
                    trajectory += 1
                    if trajectory % 50 == 0:
                        print(f"    Trajectory {trajectory} - steps = {steps_per_traj}, reward = {reward}")
                    logger.store(EpRet=ep_ret, EpLen=ep_len, ShowRet=show_ret, SJF=sjf, F1=f1)
                    batch_size = 32
                    states, actions, rewards, next_states, dones, masks = buffer.sample(batch_size)
                    agent.update(states, actions, rewards, next_states, dones, masks)

                    [state, co], reward, done, ep_ret, ep_len, show_ret, sjf, f1 = env.reset(), 0, False, 0, 0,0,0,0
                    steps_per_traj = 0
                    if trajectory >= traj_per_epoch:
                        break

            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)* traj_per_epoch * JOB_SEQUENCE_SIZE)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            # logger.log_tabular('DeltaLossPi', average_only=True)
            # logger.log_tabular('DeltaLossV', average_only=True)
            # logger.log_tabular('Entropy', average_only=True)
            # logger.log_tabular('KL', average_only=True)
            # logger.log_tabular('ClipFrac', average_only=True)
            # logger.log_tabular('StopIter', average_only=True)
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
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./data/logs/dqn_n_steps_temp/dqn_n_steps_temp_s0')
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

        train_dqn(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained=1,trained_model=os.path.join(model_file,"simple_save"),attn=args.attn,
            shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, score_type=args.score_type,
            batch_job_slice=args.batch_job_slice)
    else:
        train_dqn(workload_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained=0, attn=args.attn,shuffle=args.shuffle, backfil=args.backfil,
            skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice)