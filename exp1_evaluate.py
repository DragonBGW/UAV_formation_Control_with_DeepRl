import torch
import matplotlib.pyplot as plt

from configs.exp1_config import Exp1Config
from envs.swarm_env_exp1 import SwarmEnvExp1
from models.actor_critic import ActorCritic
import numpy as np 
cfg = Exp1Config()
env = SwarmEnvExp1(cfg)

model = ActorCritic(cfg.state_dim, cfg.action_dim).to(cfg.device)
model.load_state_dict(torch.load("results/exp1/ppo_exp1.pth"))
model.eval()

states = env.reset()

traj = []
min_dists = []
speeds = []

while True:
    traj.append(env.positions.copy())

    actions = []
    for i in range(cfg.n_uavs):
        s = torch.tensor(states[i]).float().unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            a, _, _ = model.act(s)
        actions.append(a.squeeze(0).cpu().numpy())

    states, _, done, info = env.step(np.array(actions))
    min_dists.append(info["min_dist"])
    speeds.append(info["mean_speed"])

    if done:
        break

traj = np.array(traj)

# --- Trajectory plot ---
plt.figure()
for i in range(cfg.n_uavs):
    plt.plot(traj[:, i, 0], traj[:, i, 1])
plt.title("UAV Trajectories (I → V → O → I)")
plt.show()

# --- Min distance plot ---
plt.figure()
plt.plot(min_dists)
plt.title("Minimum Inter-UAV Distance")
plt.show()

# --- Velocity plot ---
plt.figure()
plt.plot(speeds)
plt.title("Mean UAV Speed")
plt.show()
