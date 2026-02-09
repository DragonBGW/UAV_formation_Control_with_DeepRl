import torch

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment
    n_uavs = 9
    state_dim = 4
    action_dim = 2
    max_steps = 300
    vmax = 2.5
    safe_dist = 2.5

    # PPO
    gamma = 0.99
    lr = 3e-4
    clip_eps = 0.2
    epochs = 10
    batch_size = 256
    rollout_steps = 2048
