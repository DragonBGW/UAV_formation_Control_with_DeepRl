import numpy as np
from utils.formations import get_target_positions

class SwarmEnvExp3:
    def __init__(self, cfg, n_uavs):
        self.cfg = cfg
        self.n_uavs = n_uavs
        self.reset()

    def reset(self):
        self.positions = np.random.uniform(-50, 50, (self.n_uavs, 2))
        self.velocities = np.zeros((self.n_uavs, 2))
        self.target = get_target_positions("O", self.n_uavs)
        self.t = 0
        self.collision_count = 0
        return self._get_states()

    def _get_states(self):
        states = []
        for i in range(self.n_uavs):
            dists = []
            for j in range(self.n_uavs):
                if i != j:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    dists.append(d)
            dists = sorted(dists)[:4]
            states.append(dists)
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        self.velocities += actions * self.cfg.dt
        self.velocities = np.clip(self.velocities, -self.cfg.vmax, self.cfg.vmax)
        self.positions += self.velocities * self.cfg.dt

        self.t += 1
        formation_error = 0

        for i in range(self.n_uavs):
            formation_error += np.linalg.norm(
                self.positions[i] - self.target[i]
            )

            for j in range(i+1, self.n_uavs):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < self.cfg.safe_dist:
                    self.collision_count += 1

        done = (
            formation_error / self.n_uavs < self.cfg.formation_error_thresh
            or self.t >= self.cfg.max_steps
        )

        return self._get_states(), done
