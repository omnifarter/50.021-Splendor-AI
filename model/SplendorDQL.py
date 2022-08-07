import math
import random
from rl_rules import Board
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from IPython import display
import sys

################### References ##################
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#################################################



class ReplayMemory(object):

    def __init__(self, capacity):
        self.Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, input_dims, output_dims=24):
        super(DQN, self).__init__()
        
        self.linear_1 = nn.Linear(input_dims, 64) # (1, 129) x (129, 64)
        self.linear_2 = nn.Linear(64, 32) # (1, 64) x (64, 32)
        self.linear_3 = nn.Linear(32, output_dims) # (1, 32) x (32, out) = (1, out)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = torch.sigmoid(x)
        return x

class BatchData(Dataset):
    def __init__(self, data):
        # data is a list of n input states, where n is the batch size
        self.states_data = data
        
    def __len__(self):
        return len(self.states_data)

    def __getitem__(self, index):
        data = self.states_data[index]
        return data


################################## Splendor AI ###########################################

# Main class defining setup, training and model

class SplendorDQN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython: 
            from IPython import display
        plt.ion()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dims = kwargs.pop('input_dims', 129)
        self.episode_durations = []
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

        self.batch_size = kwargs.pop('batch_size', 128)
        self.gamma = kwargs.pop('gamma', 0.999)
        self.eps_start = kwargs.pop('eps_start', 0.9)
        self.eps_end = kwargs.pop('eps_end', 0.05)
        self.eps_decay = kwargs.pop('eps_decay', 200)
        self.target_update = kwargs.pop('target_update', 10)
        self.num_episodes = kwargs.pop('num_episodes', 50)

        self.n_actions = kwargs.pop('n_actions', 24) 

        # init model
        self.policy_net = DQN(self.input_dims).to(self.device)
        self.target_net = DQN(self.input_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    ############################ Training ###########################################

    def train(self):
        sys.stdout = open('../save/log.txt', 'w')
        total_wins = 0

        for i_episode in range(self.num_episodes):
            print( f"============ Game {i_episode}: Start! ================")
            # Initialize the environment and state
            board = Board()
            state = board.getState()

            for t in count():
                # Select and perform an action
                action = self._select_action(state)
                reward, done = board.playerAction(action.item())
                             
                            
                if not done:
                    # game not over, continue to next state
                    next_state =  board.getState()
                else:
                    # someone won, get reward value (sparse rewards)
                    next_state = None                    
                    if reward == 3:
                        # Player 0 won, achieved goal
                        total_wins += 1

                reward = torch.tensor([reward], device=self.device)   
                self.memory.push(state, action, next_state, reward)  # Store the transition in memory   
                state = next_state # Move to the next state

                # Perform one step of the optimization (on the policy network)
                self._optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self._plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Complete')
        self._save()
        plt.ioff()
        plt.figtext(.6, .8, f"Games Won: {total_wins} / {self.num_episodes}")
        plt.savefig(f'../save/graph.png')
        plt.show()
        sys.stdout.close()
        return
        

    def _select_action(self, state: torch.Tensor):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.reshape(state[None, :], shape=(1, 129))
                return self.policy_net(state).max(1)[1].view(-1, 1)
        else:
            return torch.tensor([[random.randrange(0, self.n_actions)]], device=self.device, dtype=torch.long)

    def _plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def _optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = [s for s in batch.next_state
                                                    if s is not None]
        non_final_loader = DataLoader(BatchData(non_final_next_states))

        state_dataset = BatchData(batch.state)
        state_batch = DataLoader(state_dataset)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        values = torch.empty((128, 24), device=self.device)
        for i, batch in enumerate(state_batch):
            values[non_final_mask] = self.policy_net(batch)
        
        state_action_values = values.gather(1, action_batch)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device) # (128,)
        for i, batch in enumerate(non_final_loader):
            next_state_values[i] = self.target_net(batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _save(self, filepath='DQL.pth'):
        torch.save(self.target_net.state_dict(), f'../save/{filepath}')

        ######################################################################
