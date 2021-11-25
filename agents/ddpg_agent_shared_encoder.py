from torch import optim

from agents.ddpg_agent import (
    Agent,
    device,
    LR_ACTOR,
    LR_CRITIC,
    WEIGHT_DECAY,
)
from models.model_shared_weights import (
    NetworkEncoder,
    ActorHead,
    CriticHead,
)


class AgentSharedEncoder(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__(state_size, action_size, random_seed)

        fc1_units = 256
        network_encoder = NetworkEncoder(state_size, random_seed, fc1_units)
        # Actor Network (w/ Target Network)

        fc2_units = 128
        self.actor_local = ActorHead(state_size, action_size, random_seed, network_encoder, fc2_units=fc2_units).to(
            device
        )
        self.actor_target = ActorHead(state_size, action_size, random_seed, network_encoder, fc2_units=fc2_units).to(
            device
        )
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = CriticHead(state_size, action_size, random_seed, network_encoder, fc2_units=fc2_units).to(
            device
        )
        self.critic_target = CriticHead(state_size, action_size, random_seed, network_encoder, fc2_units=fc2_units).to(
            device
        )
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)