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




import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# CONFIGURATION
# -------------------------------
log_dir = r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-10_10-48-51_run_multitask_analysis_moe_52\gating_weights"
num_tasks = 4
num_experts = 4
bins = np.linspace(0, 1, 21)  # histogram bins for weights

# -------------------------------
# PLOT SETUP
# -------------------------------
fig = plt.figure(figsize=(16, 12), dpi=300)
axes = [fig.add_subplot(2, 2, i+1, projection='3d') for i in range(num_tasks)]
task_names = [f"Task {t}" for t in range(num_tasks)]

# -------------------------------
# LOOP OVER TASKS
# -------------------------------
for t, ax in enumerate(axes):
    # Collect all npz files for this task
    files = sorted(glob.glob(os.path.join(log_dir, f"task_{t}_iter_*.npz")))
    if len(files) == 0:
        print(f"No data found for task {t}, skipping.")
        continue

    all_weights = []
    for f in files:
        data = np.load(f)['task_weights']  # shape: [num_samples, num_experts]
        all_weights.append(data)
    all_weights = np.concatenate(all_weights, axis=0)  # [total_samples, num_experts]

    # Prepare 3D histogram data
    xpos, ypos, dz = [], [], []

    for i in range(num_experts):
        hist, _ = np.histogram(all_weights[:, i], bins=bins)
        for j, count in enumerate(hist):
            xpos.append(i)
            ypos.append(j)
            dz.append(count)

    xpos = np.array(xpos)
    ypos = np.array(ypos)
    dz = np.array(dz)
    dx = dy = 0.6 * np.ones_like(dz)
    colors = cm.viridis(dz / dz.max() + 1e-8)

    # Plot 3D bars
    ax.bar3d(xpos, ypos, np.zeros_like(dz), dx, dy, dz, color=colors, zsort='average')

    # Axes labels and ticks
    ax.set_title(task_names[t], fontsize=12)
    ax.set_xlabel("Expert", fontsize=10)
    ax.set_ylabel("Weight Bin", fontsize=10)
    ax.set_zlabel("Count", fontsize=10)
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(len(bins)-1))
    ax.set_yticklabels([f"{b:.2f}" for b in bins[:-1]], fontsize=8)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='z', labelsize=9)

plt.tight_layout()


# -------------------------------
# PLOT 3D HISTOGRAMS FOR LAST ITERATION
# -------------------------------
fig_last, axes_last = plt.subplots(2, 2, figsize=(16, 12), dpi=300, subplot_kw={'projection':'3d'})
axes_last = axes_last.flatten()

for t, ax in enumerate(axes_last):
    # find last npz file for this task
    files = sorted(glob.glob(os.path.join(log_dir, f"task_{t}_iter_*.npz")))
    if len(files) == 0:
        print(f"No data found for task {t}, skipping last iteration histogram.")
        continue

    last_file = files[-1]
    data = np.load(last_file)['task_weights']  # [num_samples, num_experts]

    # Prepare 3D histogram data
    xpos, ypos, dz = [], [], []
    for i in range(num_experts):
        hist, _ = np.histogram(data[:, i], bins=bins)
        for j, count in enumerate(hist):
            xpos.append(i)
            ypos.append(j)
            dz.append(count)

    xpos = np.array(xpos)
    ypos = np.array(ypos)
    dz = np.array(dz)
    dx = dy = 0.6 * np.ones_like(dz)
    colors = cm.viridis(dz / dz.max() + 1e-8)

    # Plot 3D bars
    ax.bar3d(xpos, ypos, np.zeros_like(dz), dx, dy, dz, color=colors, zsort='average')

    # Axes labels and ticks
    ax.set_title(f"{task_names[t]} (last iteration)", fontsize=12)
    ax.set_xlabel("Expert", fontsize=10)
    ax.set_ylabel("Weight Bin", fontsize=10)
    ax.set_zlabel("Count", fontsize=10)
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(len(bins)-1))
    ax.set_yticklabels([f"{b:.2f}" for b in bins[:-1]], fontsize=8)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='z', labelsize=9)

plt.tight_layout()
plt.show()



import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURATION
# -------------------------------
log_dir = r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-10_10-36-56_run_multitask_analysis_52\gating_weights"

task_to_plot = 0   # e.g., Task 0
num_experts = 4

# -------------------------------
# COLLECT ALL FILES FOR THIS TASK
# -------------------------------
files = sorted(glob.glob(os.path.join(log_dir, f"task_{task_to_plot}_iter_*.npz")))
if len(files) == 0:
    raise ValueError(f"No gating weight data found for task {task_to_plot}")

iterations = []
mean_weights_per_expert = []

for f in files:
    # Extract iteration number from filename
    iter_num = int(os.path.basename(f).split("_iter_")[1].split(".npz")[0])
    iterations.append(iter_num)

    data = np.load(f)['task_weights']  # [num_samples_for_task, num_experts]
    mean_weights = data.mean(axis=0)    # mean per expert
    mean_weights_per_expert.append(mean_weights)

iterations = np.array(iterations)
mean_weights_per_expert = np.array(mean_weights_per_expert)  # [num_iterations, num_experts]

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(8, 5), dpi=300)
for i in range(num_experts):
    plt.plot(iterations, mean_weights_per_expert[:, i], label=f'Expert {i}', linewidth=2)

plt.xlabel("Training iteration", fontsize=12)
plt.ylabel("Mean gating weight", fontsize=12)
plt.title(f"Task {task_to_plot} Expert Gating Evolution", fontsize=14)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

