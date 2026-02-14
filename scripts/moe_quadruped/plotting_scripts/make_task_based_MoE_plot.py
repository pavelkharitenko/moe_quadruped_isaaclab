import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# -------------------------------
# Path to your TensorBoard log directory
# -------------------------------
log_dir = r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-10_10-48-51_run_multitask_analysis_moe_52"

# -------------------------------
# Metrics to plot
# -------------------------------
global_experts = [f"MoE/global/expert_{i}" for i in range(4)]  # adjust num_experts
task_experts = {t: [f"MoE/task_{t}/expert_{i}" for i in range(4)] for t in range(4)}
entropy_tag = "MoE/global/entropy"

# -------------------------------
# Load TensorBoard events
# -------------------------------
ea = EventAccumulator(log_dir)
ea.Reload()

# -------------------------------
# Figure style (similar to your reference)
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharey=False, dpi=300)
axes = axes.flatten()

# -------------------------------
# Plot global expert usage
# -------------------------------
for i, tag in enumerate(global_experts):
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    axes[0].plot(steps, values, linewidth=2, label=f"Expert {i}")

axes[0].set_title("Global Expert Usage")
axes[0].set_xlabel("PPO Iteration", fontsize=10)
axes[0].set_ylabel("Avg Gating Weight", fontsize=10)
axes[0].tick_params(axis='both', labelsize=10)
axes[0].grid(True)
axes[0].legend(fontsize=8)

# -------------------------------
# Plot global gating entropy
# -------------------------------
events = ea.Scalars(entropy_tag)
steps = np.array([e.step for e in events])
values = np.array([e.value for e in events])
axes[1].plot(steps, values, color="C0", linewidth=2)
axes[1].set_title("Global Gating Entropy")
axes[1].set_xlabel("PPO Iteration", fontsize=10)
axes[1].set_ylabel("Entropy", fontsize=10)
axes[1].tick_params(axis='both', labelsize=10)
axes[1].grid(True)

# -------------------------------
# Plot per-task expert usage
# -------------------------------
for t in range(2):  # tasks 0 and 1 in subplot axes[2] and axes[3], extend as needed
    ax = axes[2 + t]
    for i, tag in enumerate(task_experts[t]):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        ax.plot(steps, values, linewidth=2, label=f"Expert {i}")
    ax.set_title(f"Task {t} Expert Usage")
    ax.set_xlabel("PPO Iteration", fontsize=10)
    ax.set_ylabel("Avg Gating Weight", fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True)
    ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()