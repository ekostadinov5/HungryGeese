
from collections import deque
import numpy as np


# DQN & DDQN
class ReplayBuffer:

    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get_samples(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        indices = np.random.choice(len(self.buffer), num_samples)
        for i in indices:
            element = self.buffer[i]
            state, action, reward, next_state, done = element
            states.append(np.array(state))
            actions.append(np.array(action))
            rewards.append(reward)
            next_states.append(np.array(next_state))
            dones.append(1 if done else 0)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones


# PPO
class Memory():

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get_all_samples(self):
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions)
        rewards = np.expand_dims(np.array(self.rewards, dtype=np.float32), axis=1)
        next_states = np.array(self.next_states, dtype=np.float32)
        dones = np.expand_dims(np.array(self.dones, dtype=np.float32), axis=1)

        return states, actions, rewards, next_states, dones

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
