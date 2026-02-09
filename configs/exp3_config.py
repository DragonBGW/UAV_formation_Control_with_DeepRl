import torch

class Exp3Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Swarm sizes to test
    swarm_sizes = [25, 50, 75]

    # Environment
    state_dim = 4
    action_dim = 2
    dt = 0.1
    vmax = 2.5
    safe_dist = 2.5

    # Completion
    max_steps = 1500
    formation_error_thresh = 3.0

    # Trials
    trials_per_size = 3
