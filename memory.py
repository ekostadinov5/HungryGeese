
from collections import deque
import numpy as np


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
