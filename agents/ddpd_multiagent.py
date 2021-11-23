import random
from collections import (
    deque,
    namedtuple,
)

import numpy as np
import torch

from agents.ddpg_agent import (
    Agent,
    device,
)

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
LEARN_EVERY = 2  # Amount of steps we wait among each training
LEARN_NUMBER = 3  # Amount of times to sample the memory and train the network


class MultiAgent:
    def __init__(self, num_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.

            Params
            ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                random_seed (int): random seed
            """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.run_steps = 0

        self.agents = [Agent(state_size, action_size, random_seed) for _ in range(num_agents)]

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for s, a, r, n, d in zip(state, action, reward, next_state, done):
            self.memory.add(s, a, r, n, d)

        self.run_steps += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.run_steps % LEARN_EVERY == 0:
            for _ in range(LEARN_NUMBER):
                experiences = self.memory.sample()
                for agent in self.agents:
                    agent.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(state[i, :], add_noise))

        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = (
            torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8))
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
