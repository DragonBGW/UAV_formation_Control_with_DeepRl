import numpy as np
from utils.formations import get_target_positions
import random

class SwarmEnvExp2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.positions = np.random.uniform(-20, 20, (self.cfg.n_uavs, 2))
        self.velocities = np.zeros((self.cfg.n_uavs, 2))
        self.failed = set()
        self.t = 0

        self.target = get_target_positions("V", self.cfg.n_uavs)
        return self._get_states()

    def _inject_failure(self):
        if self.t == self.cfg.failure_time:
            self.failed = set(random.sample(
                range(self.cfg.n_uavs),
                self.cfg.num_failures
            ))
            print("⚠️ UAVs failed:", self.failed)

    def _get_states(self):
        states = []
        for i in range(self.cfg.n_uavs):
            if i in self.failed:
                states.append(np.zeros(4))
                continue

            dists = []
            for j in range(self.cfg.n_uavs):
                if i != j and j not in self.failed:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    dists.append(d)
            dists = sorted(dists)[:4]
            while len(dists) < 4:
                dists.append(100)
            states.append(dists)
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        self._inject_failure()

        for i in range(self.cfg.n_uavs):
            if i in self.failed:
                continue
            self.velocities[i] += actions[i] * self.cfg.dt
            self.velocities[i] = np.clip(
                self.velocities[i], -self.cfg.vmax, self.cfg.vmax)

        self.positions += self.velocities * self.cfg.dt

        rewards = []
        min_dist = 1e9

        for i in range(self.cfg.n_uavs):
            if i in self.failed:
                continue

            err = np.linalg.norm(self.positions[i] - self.target[i])

            for j in range(self.cfg.n_uavs):
                if i != j:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    min_dist = min(min_dist, d)
                    if d < self.cfg.safe_dist:
                        rewards.append(-1000)
                        break
            else:
                rewards.append(10.0 / (err + 1e-3))

        self.t += 1
        done = self.t >= self.cfg.max_steps

        info = {
            "min_dist": min_dist,
            "failed": list(self.failed)
        }

        return self._get_states(), np.mean(rewards), done, info
