import torch
import numpy as np

from configs.exp1_config import Exp1Config
from envs.swarm_env_exp1 import SwarmEnvExp1
from models.actor_critic import ActorCritic
from algos.ppo import PPO
from utils.buffer import RolloutBuffer

cfg = Exp1Config()
env = SwarmEnvExp1(cfg)

model = ActorCritic(cfg.state_dim, cfg.action_dim).to(cfg.device)
ppo = PPO(model, cfg)
buffer = RolloutBuffer()

episodes = 200

for ep in range(episodes):
    states = env.reset()
    ep_reward = 0

    while True:
        actions = []
        for i in range(cfg.n_uavs):
            s = torch.tensor(states[i]).float().unsqueeze(0).to(cfg.device)
            with torch.no_grad():
                a, logp, v = model.act(s)
            actions.append(a.squeeze(0).cpu().numpy())
            buffer.add(states[i], actions[-1], logp.item(), 0, False, v.item())

        next_states, reward, done, info = env.step(np.array(actions))
        buffer.rewards[-cfg.n_uavs:] = [reward] * cfg.n_uavs

        states = next_states
        ep_reward += reward

        if done:
            break

    if len(buffer.states) > cfg.rollout_steps:
        ppo.update(buffer)

    print(f"Episode {ep} | Reward {ep_reward:.2f}")

torch.save(model.state_dict(), "results/exp1/ppo_exp1.pth")
print("Model saved.")
