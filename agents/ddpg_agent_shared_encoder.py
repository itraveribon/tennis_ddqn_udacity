import random

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from agents.ddpg_agent import (
    Agent,
    device,
    LR_ACTOR,
    LR_CRITIC,
    WEIGHT_DECAY,
    OUNoise,
    EPSILON_NOISE,
    TAU,
    EPSILON_NOISE_DECAY,
)
from models.model import (
    Critic,
    Actor,
)
from models.model_shared_weights import (
    NetworkEncoder,
    ActorHead,
    CriticHead,
    ActorCriticShared,
)


# class AgentSharedEncoder:
#     """Interacts with and learns from the environment."""
#
#     def __init__(self, state_size, action_size, random_seed):
#         """Initialize an Agent object.
#
#         Params
#         ======
#             state_size (int): dimension of each state
#             action_size (int): dimension of each action
#             random_seed (int): random seed
#         """
#         self.state_size = state_size
#         self.action_size = action_size
#         self.seed = random.seed(random_seed)
#
#         # Actor Network (w/ Target Network)
#         fc1_units = 256
#         fc2_units = 128
#         self.actor_critic_local = ActorCriticShared(
#             state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units
#         ).to(device)
#         self.actor_critic_target = ActorCriticShared(
#             state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units
#         ).to(device)
#         self.actor_critic_optimizer = optim.Adam(
#             self.actor_critic_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY
#         )
#
#         # Noise process
#         self.noise = OUNoise(action_size, random_seed)
#
#         self.epsilon = EPSILON_NOISE
#
#     def act(self, state, add_noise=True):
#         """Returns actions for given state as per current policy."""
#         state = torch.from_numpy(state).float().to(device)
#         if len(state.shape) < 2:
#             dummy_actions = np.zeros(self.action_size)
#         else:
#             dummy_actions = np.zeros((state.shape[0], self.action_size))
#         dummy_actions = torch.from_numpy(dummy_actions).float().to(device)
#         self.actor_critic_local.eval()
#         with torch.no_grad():
#             action, _ = self.actor_critic_local(state, dummy_actions)
#             action = action.cpu().data.numpy()
#         self.actor_critic_local.train()
#         if add_noise:
#             action += self.noise.sample() * self.epsilon
#         return np.clip(action, -1, 1)
#
#     def reset(self):
#         self.noise.reset()
#
#     def learn(self, experiences, gamma):
#         """Update policy and value parameters using given batch of experience tuples.
#         Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
#         where:
#             actor_target(state) -> action
#             critic_target(state, action) -> Q-value
#
#         Params
#         ======
#             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
#             gamma (float): discount factor
#         """
#         states, actions, rewards, next_states, dones = experiences
#
#         actions_next, _ = self.actor_critic_target(next_states, actions)
#         _, q_targets_next = self.actor_critic_target(next_states, actions_next)
#
#         # Compute Q targets for current states (y_i)
#         q_targets = rewards + (gamma * q_targets_next * (1 - dones))
#
#         # Compute critic loss
#         actions_pred, q_expected = self.actor_critic_local(states, actions)
#         critic_loss = F.mse_loss(q_expected, q_targets)
#
#         actor_loss = -self.actor_critic_local(states, actions_pred)[1].mean()
#         # Minimize the loss
#         self.actor_critic_optimizer.zero_grad()
#         critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.actor_critic_local.parameters(), 1)
#         actor_loss.backward()
#         self.actor_critic_optimizer.step()
#
#         # ----------------------- update target networks ----------------------- #
#         self.soft_update(self.actor_critic_local, self.actor_critic_target, TAU)
#
#         self.epsilon *= EPSILON_NOISE_DECAY
#         self.noise.reset()
#
#     def soft_update(self, local_model, target_model, tau):
#         """Soft update model parameters.
#         θ_target = τ*θ_local + (1 - τ)*θ_target
#
#         Params
#         ======
#             local_model: PyTorch model (weights will be copied from)
#             target_model: PyTorch model (weights will be copied to)
#             tau (float): interpolation parameter
#         """
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# class AgentSharedEncoder(Agent):
#     """Interacts with and learns from the environment."""
#
#     def __init__(self, state_size, action_size, random_seed):
#         """Initialize an Agent object.
#
#         Params
#         ======
#             state_size (int): dimension of each state
#             action_size (int): dimension of each action
#             random_seed (int): random seed
#         """
#         super().__init__(state_size, action_size, random_seed)
#
#         fc1_units = 256
#         network_encoder = NetworkEncoder(state_size, random_seed, fc1_units)
#         # Actor Network (w/ Target Network)
#
#         fc2_units = 128
#         self.actor_local = ActorHead(state_size, action_size, random_seed, network_encoder, fc2_units=fc2_units).to(
#             device
#         )
#         self.actor_target = ActorHead(state_size, action_size, random_seed, network_encoder, fc2_units=fc2_units).to(
#             device
#         )
#         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
#
#         # Critic Network (w/ Target Network)
#         self.critic_local = CriticHead(state_size, action_size, random_seed, network_encoder, fc2_units=fc2_units).to(
#             device
#         )
#         self.critic_target = CriticHead(state_size, action_size, random_seed, network_encoder, fc2_units=fc2_units).to(
#             device
#         )
#         self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)


class AgentSharedEncoder:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
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

        fc1_units = 256
        self.network_encoder_local = NetworkEncoder(state_size, random_seed, fc1_units).to(device)
        self.network_encoder_target = NetworkEncoder(state_size, random_seed, fc1_units).to(device)
        # Actor Network (w/ Target Network)

        fc2_units = 128
        self.actor_local = ActorHead(fc1_units, action_size, random_seed, fc2_units=fc2_units).to(device)
        self.actor_target = ActorHead(fc1_units, action_size, random_seed, fc2_units=fc2_units).to(device)
        self.actor_optimizer = optim.Adam(
            list(self.actor_local.parameters()) + list(self.network_encoder_local.parameters()), lr=LR_ACTOR
        )

        # Critic Network (w/ Target Network)
        self.critic_local = CriticHead(fc1_units, action_size, random_seed, fc2_units=fc2_units).to(device)
        self.critic_target = CriticHead(fc1_units, action_size, random_seed, fc2_units=fc2_units).to(device)

        self.critic_optimizer = optim.Adam(
            list(self.critic_local.parameters()) + list(self.network_encoder_local.parameters()),
            lr=LR_CRITIC,
            weight_decay=WEIGHT_DECAY,
        )

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.epsilon = EPSILON_NOISE

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.network_encoder_local.eval()
        self.actor_local.eval()
        with torch.no_grad():
            state_encoded = self.network_encoder_local(state)
            action = self.actor_local(state_encoded).cpu().data.numpy()
        self.actor_local.train()
        self.network_encoder_local.train()
        if add_noise:
            action += self.noise.sample() * self.epsilon
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_states_encoded = self.network_encoder_target(next_states)
        actions_next = self.actor_target(next_states_encoded)
        q_targets_next = self.critic_target(next_states_encoded, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        states_encoded = self.network_encoder_local(states)
        q_expected = self.critic_local(states_encoded, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        states_encoded = self.network_encoder_local(states)
        actions_pred = self.actor_local(states_encoded)
        actor_loss = -self.critic_local(states_encoded, actions_pred).mean()
        # Minimize the loss()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.network_encoder_local, self.network_encoder_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        self.epsilon *= EPSILON_NOISE_DECAY
        self.noise.reset()

    def reset(self):
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
