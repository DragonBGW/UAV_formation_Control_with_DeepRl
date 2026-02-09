import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, s, a, logp, r, d, v):
        self.states.append(s)
        self.actions.append(a)
        self.logprobs.append(logp)
        self.rewards.append(r)
        self.dones.append(d)
        self.values.append(v)

    def compute_returns(self, gamma=0.99):
        returns = []
        G = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                G = 0
            G = r + gamma * G
            returns.insert(0, G)
        return np.array(returns)
