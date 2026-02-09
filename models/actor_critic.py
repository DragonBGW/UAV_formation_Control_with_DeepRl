import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(state)

        return action, logprob, value

    def evaluate(self, states, actions):
        mean = self.actor(states)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        logprobs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states).squeeze(-1)

        return logprobs, entropy, values
