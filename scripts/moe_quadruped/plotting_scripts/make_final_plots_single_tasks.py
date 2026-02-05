import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ------------------------------------------------------------------
# Experiments: each entry is a list of seed directories
# ------------------------------------------------------------------
experiments = {
    "Task 1: Velocity Tracking": [
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_07-10-54_run_flatvel_final_53",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_08-04-53_run_flatvel_final_449",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_08-44-56_run_flatvel_final_12",
    ],
    "Task 2: Handstand": [
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_09-39-04_run_handstand_final_53",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_10-45-59_run_handstand_final_449",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_11-33-24_run_handstand_final_12",
    ],
    "Task 3: Legstand": [
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_13-49-23_run_legstand_final_52",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_14-46-27_run_legstand_final_449",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_15-38-34_run_legstand_final_12",
    ],
    "Task 4: Crawl": [
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_16-59-34_run_lowcrawl_final_52",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_17-58-21_run_lowcrawl_final_449",
    r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\final_runs\2026-02-02_18-59-17_run_lowcrawl_final_12",
    ],
}

reward_tag = "Train/mean_reward"  # PPO iterations
# reward_tag = "Train/mean_reward/time"  # environment steps

# ------------------------------------------------------------------
# Create figure (2x2) and set higher DPI
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharey=False, dpi=300)
axes = axes.flatten()

# ------------------------------------------------------------------
# Loop over experiments
# ------------------------------------------------------------------
for ax, (exp_name, seed_dirs) in zip(axes, experiments.items()):

    all_rewards = []
    all_steps = []

    for log_dir in seed_dirs:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        events = event_acc.Scalars(reward_tag)

        steps = np.array([e.step for e in events])
        rewards = np.array([e.value for e in events])

        all_steps.append(steps)
        all_rewards.append(rewards)

    # --------------------------------------------------------------
    # Align seeds (truncate to shortest)
    # --------------------------------------------------------------
    min_len = min(len(r) for r in all_rewards)
    all_rewards = np.array([r[:min_len] for r in all_rewards])
    steps = all_steps[0][:min_len]

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------
    mean_reward = np.mean(all_rewards, axis=0)
    std_reward = np.std(all_rewards, axis=0)

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    ax.plot(steps, mean_reward, linewidth=2)
    ax.fill_between(
        steps,
        mean_reward - std_reward,
        mean_reward + std_reward,
        alpha=0.3
    )

    ax.set_title(exp_name)
    ax.set_xlabel("iteration steps", fontsize=10)  # increased font size
    ax.set_ylabel("Episode Return", fontsize=10)   # increased font size (redundant only for left panels)
    ax.tick_params(axis='both', labelsize=10)      # larger ticks
    ax.grid(True)

# ------------------------------------------------------------------
# Shared y-label only for first column
# ------------------------------------------------------------------
# axes[0].set_ylabel("Episode Return")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
