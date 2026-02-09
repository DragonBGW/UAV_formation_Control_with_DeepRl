# UAV_formation_Control_with_DeepRL
Activation - 
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

A. Experiment 1 - 
Formation Reconstruction (I -> V -> O -> I) 
python exp1_train.py
model saved to results/exp1/ppo_exp1.pth
evaluation -> python exp1_evaluate.py
<img width="477" height="408" alt="image" src="https://github.com/user-attachments/assets/5e344aba-adc2-4792-b4fd-3ea857759086" />
<img width="481" height="406" alt="image" src="https://github.com/user-attachments/assets/d95f3e23-1064-48b2-8785-ae0ddd7ef020" />
<img width="482" height="411" alt="image" src="https://github.com/user-attachments/assets/0cb42d1a-f3e1-46c9-bdee-fbb9a021baea" />


📊 Graphs Generated
1️⃣ UAV Trajectories (I → V → O → I)
What it shows:

Each colored line = one UAV’s motion over time.

The shape bending represents formation transitions.

Smooth curves indicate stable control.

What it validates:

Successful formation switching.

Coordinated multi-agent motion.

Absence of chaotic oscillation.

If trajectories:

Are smooth → controller is stable.

Diverge wildly → policy unstable.

Collapse together → collision issue.

2️⃣ Minimum Inter-UAV Distance vs Time
What it shows:
Smallest pairwise distance in swarm at each timestep.

What it validates:
Safety constraint enforcement.
Collision avoidance performance.

If curve:
Stays above safe threshold → safe behavior.
Drops sharply → collision occurred.
Oscillates heavily → unstable formation.

3️⃣ Mean UAV Speed vs Time
What it shows:
Average velocity magnitude of swarm over time.

What it validates:
Smooth velocity transitions.
No aggressive acceleration spikes.
Stability during formation switching.

If curve:
Smooth → good motion planning.
Spiky → unstable control.
Flat near zero → stagnation.

🧪 Experiment 2 — Failure Injection & Recovery
python exp2_train.py
model saved to results/exp2/ppo_exp2.pth
running evaluation python exp2_evaluate.py
<img width="482" height="451" alt="image" src="https://github.com/user-attachments/assets/ddec267b-96e8-4fd9-a997-177d5c37ae3c" />
<img width="482" height="410" alt="image" src="https://github.com/user-attachments/assets/e490567a-3276-497e-8047-aa464371d78f" />


📊 Graphs Generated
1️⃣ UAV Recovery Trajectories (Failure Injection)
What it shows:
Each line = one UAV.
Trajectory bending after failure time.
Failed UAVs stop moving.

What it validates:
Autonomous fault detection.
Dynamic trajectory re-planning.
Swarm robustness.

If curves:
Bend and stabilize → recovery working.
Collapse → unsafe recovery.
Diverge → unstable response.

2️⃣ Minimum Distance During Failure Recovery
Includes:
Red vertical line = failure injection time.

What it shows:
Safety before and after failure.

What it validates:
Whether swarm avoids failed UAVs.
Whether recovery induces collisions.

If curve:
Stable after failure → robust.
Drops sharply → collision due to disturbance.

🧪 Experiment 3 — Large-Scale Swarm Benchmark
Tests scalability on 
25 UAVs
50 UAVs
75 UAVs

running benchmark -> python exp3_benchmark.py
<img width="403" height="208" alt="image" src="https://github.com/user-attachments/assets/ac7944eb-22c3-4b30-a28d-77722499956b" />
<img width="529" height="142" alt="image" src="https://github.com/user-attachments/assets/a9508ebe-f2ce-45b6-a984-19dd7552899a" />

📊 Metrics Collected
1️⃣ Completion Time

What it measures:
Time required to reach formation.
Computational + convergence performance.

Expected behavior:
Time increases with swarm size.
Ideally linear growth.

If time:
Explodes exponentially → scalability issue.
Increases smoothly → acceptable scaling.

2️⃣ Collision Count

What it measures:
Total number of safety violations.
Quality of policy generalization to larger swarms.

Expected behavior (paper):
Near zero collisions.

Observed in your reproduction:
Collisions increase with swarm size.

This indicates:
Policy trained on small swarm does not generalize perfectly.
Hierarchical control (used in paper) improves this.

| Experiment | What It Tests            | What Success Looks Like               |
| ---------- | ------------------------ | ------------------------------------- |
| Exp-1      | Formation reconstruction | Smooth trajectories + stable distance |
| Exp-2      | Robustness to failure    | Recovery without collapse             |
| Exp-3      | Scalability              | Moderate time growth + low collisions |







