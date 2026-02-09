import torch
import torch.nn as nn
import numpy as np

class PPO:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def update(self, buffer):
        states = torch.tensor(buffer.states).float().to(self.cfg.device)
        actions = torch.tensor(buffer.actions).float().to(self.cfg.device)
        old_logprobs = torch.tensor(buffer.logprobs).float().to(self.cfg.device)

        returns = buffer.compute_returns(self.cfg.gamma)
        returns = torch.tensor(returns).float().to(self.cfg.device)

        for _ in range(self.cfg.epochs):
            logprobs, entropy, values = self.model.evaluate(states, actions)

            advantages = returns - values.detach()
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios, 
                1 - self.cfg.clip_eps, 
                1 + self.cfg.clip_eps
            ) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        buffer.clear()
