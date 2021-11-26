import torch
from torch import nn

from models.model import (
    hidden_init,
    Actor,
    Critic,
)
import torch.nn.functional as F


class ActorCriticShared(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
               Params
               ======
                   state_size (int): Dimension of each state
                   action_size (int): Dimension of each action
                   seed (int): Random seed
                   fc1_units (int): Number of nodes in first hidden layer
                   fc2_units (int): Number of nodes in second hidden layer
               """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2_actor = nn.Linear(fc1_units, fc2_units)
        self.fc3_actor = nn.Linear(fc2_units, action_size)
        self.sequential_actor = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.BatchNorm1d(fc1_units),
            self.fc2_actor,
            nn.ReLU(),
            nn.BatchNorm1d(fc2_units),
            self.fc3_actor,
            nn.Tanh(),
        )

        self.fc2_critic = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3_critic = nn.Linear(fc2_units, 1)
        self.sequential_critic = nn.Sequential(self.fc1, nn.ReLU(), nn.BatchNorm1d(fc1_units))

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

        self.fc2_actor.weight.data.uniform_(*hidden_init(self.fc2_actor))
        self.fc3_actor.weight.data.uniform_(-3e-3, 3e-3)

        self.fc2_critic.weight.data.uniform_(*hidden_init(self.fc2_critic))
        self.fc3_critic.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build an actor (policy) network that maps states -> actions."""
        if len(state.size()) < 2:
            state = state[None, :]
            action = action[None, :]

        x_actor = self.sequential_actor(state)

        xs = self.sequential_critic(state)
        x_critic = torch.cat((xs, action), dim=1)
        x_critic = F.relu(self.fc2_critic(x_critic))
        x_critic = self.fc3_critic(x_critic)

        return x_actor, x_critic


class NetworkEncoder(nn.Module):
    def __init__(self, state_size, seed, fc1_units=400):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.sequential = nn.Sequential(self.fc1, nn.ReLU(), nn.BatchNorm1d(fc1_units))

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

    def forward(self, state):
        if len(state.size()) < 2:
            state = state[None, :]
        return self.sequential(state)


class ActorHead(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size_encoded, action_size, seed, fc2_units=300):
        """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
                fc1_units (int): Number of nodes in first hidden layer
                fc2_units (int): Number of nodes in second hidden layer
            """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc2 = nn.Linear(state_size_encoded, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.sequential = nn.Sequential(self.fc2, nn.ReLU(), nn.BatchNorm1d(fc2_units), self.fc3, nn.Tanh())
        self.reset_parameters()

    def reset_parameters(self):
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if len(state.size()) < 2:
            state = state[None, :]
        x = self.sequential(state)
        return x


class CriticHead(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size_encoded, action_size, seed, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc2 = nn.Linear(state_size_encoded + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_encoded, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state_encoded, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
