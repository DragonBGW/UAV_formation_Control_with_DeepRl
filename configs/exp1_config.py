import torch

class Exp1Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Swarm
    n_uavs = 9
    state_dim = 4
    action_dim = 2
    dt = 0.1
    vmax = 2.5
    safe_dist = 2.5

    # Formation timing
    formation_steps = 200
    total_steps = formation_steps * 4

    # PPO
    gamma = 0.99
    lr = 3e-4
    clip_eps = 0.2
    epochs = 10
    rollout_steps = 2048
