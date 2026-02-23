import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# -------------------------------
# CONFIGURATION
# -------------------------------
log_dir = r"logs\rsl_rl\unitree_go2_multitask\2026-02-10_10-48-51_run_multitask_analysis_moe_52\gating_weights"

num_tasks = 4
num_experts = 4

# -------------------------------
# FIGURE + AXES
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(6, 3), dpi=300)
axes = axes.flatten()

# Leave space on the right for colorbar
plt.subplots_adjust(right=0.85, hspace=0.55, wspace=0.3)

task_titles = [
    "FlatVel",
    "Handstand",
    "Legstand",
    "Crawl",
]
# -------------------------------
# COLOR SETTINGS
# -------------------------------
cmap = cm.viridis
norm = Normalize(vmin=0.0, vmax=0.7)

# -------------------------------
# LOOP OVER TASKS
# -------------------------------
for t, ax in enumerate(axes):
    files = sorted(glob.glob(os.path.join(log_dir, f"task_{t}_iter_*.npz")))

    if len(files) == 0:
        print(f"No data found for task {t}")
        continue

    all_weights = []
    for f in files:
        data = np.load(f)['task_weights']
        all_weights.append(data)

    all_weights = np.concatenate(all_weights, axis=0)
    mean_weights = all_weights.mean(axis=0)

    colors = cmap(norm(mean_weights))

    experts = np.arange(num_experts)
    ax.bar(experts, mean_weights, color=colors)

    # ---- Style matching your plots ----
    ax.set_title(f"Task {t}: {task_titles[t]}", fontsize=10)
    ax.set_xlabel("Expert", fontsize=10)
    ax.set_ylabel("Mean Gating Weight", fontsize=10)

    ax.set_ylim(0, 0.75)
    ax.set_xticks(experts)

    ax.tick_params(axis="both", labelsize=10)
    #ax.grid(True)

# -------------------------------
# SIDE COLORBAR
# -------------------------------

cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax)
#cbar.set_label("Mean Weight", fontsize=10)
cbar.ax.tick_params(labelsize=10)

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# -------------------------------
# CONFIGURATION
# -------------------------------
log_dir = r"logs\rsl_rl\unitree_go2_multitask\2026-02-10_10-48-51_run_multitask_analysis_moe_52\gating_weights"

num_tasks = 4
num_experts = 4

task_titles = [
    "FlatVel",
    "Handstand",
    "Legstand",
    "Crawl",
]

# -------------------------------
# FIGURE + AXES
# -------------------------------
fig, axes = plt.subplots(2, 2, figsize=(6, 3), dpi=300)
axes = axes.flatten()
plt.subplots_adjust(right=0.85, hspace=0.55, wspace=0.3)

# -------------------------------
# COLOR SETTINGS
# -------------------------------
cmap = cm.viridis

# -------------------------------
# FIRST PASS: compute all σ for normalization
# -------------------------------
all_std = []

for t in range(num_tasks):
    files = sorted(glob.glob(os.path.join(log_dir, f"task_{t}_iter_*.npz")))
    if len(files) == 0:
        continue

    all_weights = []
    for f in files:
        data = np.load(f)['task_weights']
        all_weights.append(data)
    all_weights = np.concatenate(all_weights, axis=0)

    var_weights = all_weights.var(axis=0)  # variance per expert
    std_weights = np.sqrt(var_weights)  # std deviation σ
    all_std.append(std_weights)

# Global max σ across all tasks for color scaling
global_max_std = np.max(np.array(all_std))
norm = Normalize(vmin=0.0, vmax=global_max_std)

# -------------------------------
# LOOP OVER TASKS TO PLOT
# -------------------------------
for t, ax in enumerate(axes):
    files = sorted(glob.glob(os.path.join(log_dir, f"task_{t}_iter_*.npz")))
    if len(files) == 0:
        print(f"No data found for task {t}")
        continue

    all_weights = []
    for f in files:
        data = np.load(f)['task_weights']
        all_weights.append(data)
    all_weights = np.concatenate(all_weights, axis=0)

    # Mean and σ per expert
    mean_weights = all_weights.mean(axis=0)
    std_weights = np.sqrt(all_weights.var(axis=0))

    # Bar colors based on σ
    colors = cmap(norm(std_weights))

    # -------------------------------
    # Plot bars
    # -------------------------------
    experts = np.arange(num_experts)
    bars = ax.bar(experts, mean_weights, color=colors, zorder=2)  # zorder>1 to put bars above grid

    # -------------------------------
    # Styling
    # -------------------------------
    ax.set_title(f"Task {t}: {task_titles[t]}", fontsize=10)
    ax.set_xlabel("Expert", fontsize=10)
    ax.set_ylabel("Mean Gating Weight", fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.set_xticks(experts)
    ax.tick_params(axis="both", labelsize=10)

    # Grid behind bars
    ax.grid(True, zorder=0)

# -------------------------------
# SIDE COLORBAR (σ)
# -------------------------------
cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Std. Dev. (σ) of Gating Weight", fontsize=10)
cbar.ax.tick_params(labelsize=10)

plt.show()

plt.show()
