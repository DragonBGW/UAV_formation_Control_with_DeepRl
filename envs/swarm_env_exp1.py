import numpy as np
from utils.formations import get_target_positions

class SwarmEnvExp1:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.positions = np.random.uniform(-20, 20, (self.cfg.n_uavs, 2))
        self.velocities = np.zeros((self.cfg.n_uavs, 2))
        self.t = 0
        self.formation_seq = ["I", "V", "O", "I"]
        self.current_target = get_target_positions("I", self.cfg.n_uavs)
        return self._get_states()

    def _update_target(self):
        idx = self.t // self.cfg.formation_steps
        name = self.formation_seq[min(idx, 3)]
        self.current_target = get_target_positions(name, self.cfg.n_uavs)

    def _get_states(self):
        states = []
        for i in range(self.cfg.n_uavs):
            dists = []
            for j in range(self.cfg.n_uavs):
                if i != j:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    dists.append(d)
            dists = sorted(dists)[:4]
            states.append(dists)
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        self._update_target()

        # apply actions
        self.velocities += actions * self.cfg.dt
        self.velocities = np.clip(self.velocities, -self.cfg.vmax, self.cfg.vmax)
        self.positions += self.velocities * self.cfg.dt

        rewards = []
        min_dist = 1e9

        for i in range(self.cfg.n_uavs):
            cur = self.positions[i]
            tgt = self.current_target[i]
            err = np.linalg.norm(cur - tgt)

            # collision check
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
        done = self.t >= self.cfg.total_steps

        info = {
            "min_dist": min_dist,
            "mean_speed": np.mean(np.linalg.norm(self.velocities, axis=1))
        }

        return self._get_states(), np.mean(rewards), done, info
