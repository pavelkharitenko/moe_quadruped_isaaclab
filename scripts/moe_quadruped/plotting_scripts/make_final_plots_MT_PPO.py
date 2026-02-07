import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ------------------------------------------------------------------
# Multitask runs (different seeds)
# ------------------------------------------------------------------
run_dirs = [
    #r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-04_09-30-23_run_multitask_exp_ppo_449",
    #r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-04_14-03-44_run_multitask_final_ppo_12"
    
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-05_17-28-18_run_multitask_final_moe_449",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-06_08-29-39_run_multitask_final_moe_12",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-07_07-55-31_run_multitask_final_moe_52",
]

# ------------------------------------------------------------------
# Tags
# ------------------------------------------------------------------
overall_tag = "Train/mean_reward"

task_tags = {
    "FlatVel": "Rewards/flatVel/mean_per_episode",
    "Handstand": "Rewards/handstand/mean_per_episode",
    "Legstand": "Rewards/legStand/mean_per_episode",
    "Crawl": "Rewards/lowcrawl/mean_per_episode",
}

seaborn_colors = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B2",
    "brown":  "#937860",
    "pink":   "#DA8BC3",
    "gray":   "#8C8C8C",
    "yellow": "#CCB974",
    "cyan":   "#64B5CD",
}

task_colors = {
    "FlatVel": "#64B5CD",
    "Handstand": "#55A868",
    "Legstand": "#DD8452",
    "Crawl": "#8172B2",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def forward_fill_nan(x):
    out = []
    last = 0.0
    for v in x:
        if np.isnan(v):
            out.append(last)
        else:
            out.append(v)
            last = v
    return np.array(out)

def running_average(x, window=40):
    if window <= 1:
        return x

    pad = window // 2
    x_padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(x_padded, kernel, mode="valid")

    return smoothed[:len(x)]



# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------
all_overall = []
all_tasks = {k: [] for k in task_tags}

# ------------------------------------------------------------------
# Load runs
# ------------------------------------------------------------------
for run_dir in run_dirs:
    event_acc = EventAccumulator(run_dir)
    event_acc.Reload()

    # Overall reward
    events = event_acc.Scalars(overall_tag)
    steps = np.array([e.step for e in events])
    rewards = np.array([e.value for e in events])
    rewards = running_average(rewards)
    all_overall.append(rewards)

    # Per-task rewards
    for task, tag in task_tags.items():
        events = event_acc.Scalars(tag)
        r = np.array([e.value for e in events])
        r = forward_fill_nan(r)
        r = running_average(r)
        all_tasks[task].append(r)

# ------------------------------------------------------------------
# Align lengths
# ------------------------------------------------------------------
min_len = min(len(r) for r in all_overall)
steps = steps[:min_len]

all_overall = np.array([r[:min_len] for r in all_overall])
for task in all_tasks:
    all_tasks[task] = np.array([r[:min_len] for r in all_tasks[task]])

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
overall_mean = np.mean(all_overall, axis=0)
overall_std = np.std(all_overall, axis=0)

task_mean = {k: np.mean(v, axis=0) for k, v in all_tasks.items()}
task_std  = {k: np.std(v, axis=0)  for k, v in all_tasks.items()}

# ------------------------------------------------------------------
# Plot: 2x1 layout
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(4, 5), dpi=300, sharex=True)

# ==============================
# Top: Overall mean reward
# ==============================
ax = axes[0]
ax.plot(steps, overall_mean, linewidth=2, label="Overall mean reward")
ax.fill_between(
    steps,
    overall_mean - overall_std,
    overall_mean + overall_std,
    alpha=0.25,
)

ax.set_ylabel("Episode Return", fontsize=10)
ax.tick_params(axis="both", labelsize=10)
ax.grid(True)
ax.legend(fontsize=9)

# ==============================
# Bottom: Per-task rewards
# ==============================
ax = axes[1]
for task in task_tags:
    ax.plot(
        steps,
        task_mean[task],
        linewidth=2,
        label=task,
        color=task_colors[task],
    )
    ax.fill_between(
        steps,
        task_mean[task] - task_std[task],
        task_mean[task] + task_std[task],
        alpha=0.25,
        color=task_colors[task],
    )

ax.set_xlabel("iteration steps", fontsize=10)
ax.set_ylabel("Episode Return", fontsize=10)
ax.tick_params(axis="both", labelsize=10)
ax.grid(True)
ax.legend(fontsize=9, ncol=2)

plt.tight_layout()
plt.show()
