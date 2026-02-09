import torch
import numpy as np
import matplotlib.pyplot as plt

from configs.exp2_config import Exp2Config
from envs.swarm_env_exp2 import SwarmEnvExp2
from models.actor_critic import ActorCritic

cfg = Exp2Config()
env = SwarmEnvExp2(cfg)

model = ActorCritic(cfg.state_dim, cfg.action_dim).to(cfg.device)
model.load_state_dict(torch.load("results/exp2/ppo_exp2.pth"))
model.eval()

states = env.reset()
traj = []
failed_log = []
min_dists = []

while True:
    traj.append(env.positions.copy())
    failed_log.append(list(env.failed))

    actions = []
    for i in range(cfg.n_uavs):
        s = torch.tensor(states[i]).float().unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            a, _, _ = model.act(s)
        actions.append(a.squeeze(0).cpu().numpy())

    states, _, done, info = env.step(np.array(actions))
    min_dists.append(info["min_dist"])

    if done:
        break

traj = np.array(traj)

# --- Trajectory plot ---
plt.figure()
for i in range(cfg.n_uavs):
    plt.plot(traj[:, i, 0], traj[:, i, 1])
plt.title("UAV Recovery Trajectories (Failure Injection)")
plt.show()

# --- Min distance plot ---
plt.figure()
plt.plot(min_dists)
plt.axvline(cfg.failure_time, color="r", linestyle="--", label="Failure")
plt.legend()
plt.title("Minimum Distance During Failure Recovery")
plt.show()
