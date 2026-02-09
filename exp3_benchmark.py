import time
import torch
import numpy as np

from configs.exp3_config import Exp3Config
from envs.swarm_env_exp3 import SwarmEnvExp3
from models.actor_critic import ActorCritic

cfg = Exp3Config()

model = ActorCritic(cfg.state_dim, cfg.action_dim).to(cfg.device)
model.load_state_dict(torch.load("results/exp1/ppo_exp1.pth"))
model.eval()

results = []

for n_uavs in cfg.swarm_sizes:
    print(f"\n🚀 Benchmarking {n_uavs} UAVs")

    for trial in range(cfg.trials_per_size):
        env = SwarmEnvExp3(cfg, n_uavs)
        states = env.reset()

        start_time = time.time()

        while True:
            actions = []
            for i in range(n_uavs):
                s = torch.tensor(states[i]).float().unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    a, _, _ = model.act(s)
                actions.append(a.squeeze(0).cpu().numpy())

            states, done = env.step(np.array(actions))
            if done:
                break

        elapsed = time.time() - start_time

        print(f"Trial {trial+1} | Time: {elapsed:.2f}s | Collisions: {env.collision_count}")

        results.append({
            "n_uavs": n_uavs,
            "trial": trial,
            "time": elapsed,
            "collisions": env.collision_count
        })

print("\n✅ Benchmark complete.")
print(results)
