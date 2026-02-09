import torch
from configs.config import Config
from envs.swarm_env import SwarmEnv
from models.actor_critic import ActorCritic
from algos.ppo import PPO
from utils.buffer import RolloutBuffer

cfg = Config()
env = SwarmEnv(cfg)

model = ActorCritic(cfg.state_dim, cfg.action_dim).to(cfg.device)
ppo = PPO(model, cfg)
buffer = RolloutBuffer()

for episode in range(300):
    state = env.reset()
    ep_reward = 0

    while True:
        state_t = torch.tensor(state).float().unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            action, logp, value = model.act(state_t)

        action_np = action.squeeze(0).cpu().numpy()
        next_state, reward, done, _ = env.step(action_np)

        buffer.add(
            state, 
            action_np, 
            logp.item(), 
            reward, 
            done, 
            value.item()
        )

        state = next_state
        ep_reward += reward

        if done:
            break

    print(f"Episode {episode} | Reward: {ep_reward:.2f}")

    if len(buffer.states) > cfg.rollout_steps:
        ppo.update(buffer)

torch.save(model.state_dict(), "ppo_swarm.pth")
print("Model saved.")
