import torch
from torch import nn

from models.model import (
    hidden_init,
    Actor,
    Critic,
)


class NetworkEncoder(nn.Module):
    def __init__(self, state_size, seed, fc1_units=400):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

    def forward(self, state):
        return self.fc1(state)

    @property
    def out_features(self):
        return self.fc1.out_features


class ActorHead(Actor):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, network_encoder, fc2_units=300):
        """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
                fc1_units (int): Number of nodes in first hidden layer
                fc2_units (int): Number of nodes in second hidden layer
            """
        super().__init__(state_size, action_size, seed, network_encoder.out_features, fc2_units)
        self.fc1 = network_encoder

        self.sequential = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.BatchNorm1d(network_encoder.out_features),
            self.fc2,
            nn.ReLU(),
            nn.BatchNorm1d(fc2_units),
            self.fc3,
            nn.Tanh(),
        )

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class CriticHead(Critic):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, network_encoder, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__(state_size, action_size, seed, network_encoder.out_features, fc2_units)
        self.fcs1 = network_encoder

        self.sequential = nn.Sequential(self.fcs1, nn.ReLU(), nn.BatchNorm1d(network_encoder.out_features))

    def reset_parameters(self):
        self.fcs1.reset_parameters()
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
