import numpy as np

class SwarmEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.positions = np.random.uniform(-5, 5, (self.cfg.n_uavs, 2))
        self.velocities = np.zeros((self.cfg.n_uavs, 2))
        self.t = 0
        return self._get_state(0)

    def _get_state(self, i):
        dists = []
        for j in range(self.cfg.n_uavs):
            if i != j:
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                dists.append(d)
        dists = sorted(dists)[:4]
        return np.array(dists, dtype=np.float32)

    def step(self, action):
        self.velocities[0] += action * 0.1
        self.velocities[0] = np.clip(
            self.velocities[0], 
            -self.cfg.vmax, 
            self.cfg.vmax
        )

        self.positions += self.velocities * 0.1
        self.t += 1

        state = self._get_state(0)

        # reward
        target = np.ones(4) * 5.0
        diff = state - target
        if np.any(diff < -self.cfg.safe_dist):
            reward = -1000
            done = True
        else:
            reward = 4.0 / (np.sum(diff**2) + 1e-6)
            done = False

        if self.t >= self.cfg.max_steps:
            done = True

        return state, reward, done, {}
