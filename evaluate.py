import torch
import matplotlib.pyplot as plt
from configs.config import Config
from envs.swarm_env import SwarmEnv
from models.actor_critic import ActorCritic

cfg = Config()
env = SwarmEnv(cfg)

model = ActorCritic(cfg.state_dim, cfg.action_dim).to(cfg.device)
model.load_state_dict(torch.load("ppo_swarm.pth"))
model.eval()

state = env.reset()
positions = []

while True:
    positions.append(env.positions.copy())

    state_t = torch.tensor(state).float().unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        action, _, _ = model.act(state_t)

    action = action.squeeze(0).cpu().numpy()
    state, _, done, _ = env.step(action)

    if done:
        break

positions = positions[-1]
plt.scatter(positions[:,0], positions[:,1])
plt.title("Final UAV Positions")
plt.show()
