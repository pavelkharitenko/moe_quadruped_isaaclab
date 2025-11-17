import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def analyze_tfevents_multitask(event_file: str, max_extra_plots: int = 8):
    """
    TensorBoard analyzer for multitask IsaacLab runs.

    Detects:
      - overall mean reward (Train/mean_reward)
      - per-task per-step rewards (Rewards/<task>/mean_per_step)
      - per-task episode rewards (Rewards/<task>/mean_per_episode)
      - per-task episode lengths (Rewards/<task>/mean_episode_length)
    """

    if not os.path.exists(event_file):
        raise FileNotFoundError(f"No such file: {event_file}")

    print(f"[INFO] Loading event file: {event_file}")
    ea = EventAccumulator(event_file)
    ea.Reload()

    scalar_tags = ea.Tags()["scalars"]
    print(f"\n[INFO] Found {len(scalar_tags)} scalar tags:\n")
    for t in scalar_tags:
        print("   ", t)
    print()

    # --------------------------------------------
    # Identify tag groups
    # --------------------------------------------
    total_reward_tags = [t for t in scalar_tags if t == "Train/mean_reward"]

    step_tags = [t for t in scalar_tags if re.match(r"Rewards/.+/mean_per_step$", t)]
    episode_tags = [t for t in scalar_tags if re.match(r"Rewards/.+/mean_per_episode$", t)]
    ep_len_tags = [t for t in scalar_tags if re.match(r"Rewards/.+/mean_episode_length$", t)]

    task_names = sorted(list(set([t.split("/")[1] for t in episode_tags + ep_len_tags + step_tags])))

    # --------------------------------------------
    # Create layout
    # --------------------------------------------
    n_tasks = len(task_names)
    n_rows = n_tasks + 1  # total + per-task
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # --- Total mean reward ---
    ax = axes[0]
    if total_reward_tags:
        tag = total_reward_tags[0]
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        ax.plot(steps, values, color="tab:blue", label="Train/mean_reward")
        print(f"[INFO] {tag}: mean={values.mean():.2f}, max={values.max():.2f}, final={values[-1]:.2f}")
        ax.set_title("Train/mean_reward (total)")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        ax.set_visible(False)

    # --------------------------------------------
    # Per-task plots
    # --------------------------------------------
    for i, task_name in enumerate(task_names, start=1):
        ax = axes[i]
        ax.set_title(f"{task_name} rewards")
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Reward / Length")

        # per-step reward
        step_tag = f"Rewards/{task_name}/mean_per_step"
        if step_tag in scalar_tags:
            events = ea.Scalars(step_tag)
            steps = np.array([e.step for e in events])
            vals = np.array([e.value for e in events])
            ax.plot(steps, vals, label="Per-step mean", color="tab:orange")

        # per-episode reward
        ep_tag = f"Rewards/{task_name}/mean_per_episode"
        if ep_tag in scalar_tags:
            events = ea.Scalars(ep_tag)
            steps = np.array([e.step for e in events])
            vals = np.array([e.value for e in events])
            ax.plot(steps, vals, label="Per-episode mean", color="tab:blue")

        # episode length
        ep_len_tag = f"Rewards/{task_name}/mean_episode_length"
        if ep_len_tag in scalar_tags:
            events = ea.Scalars(ep_len_tag)
            steps = np.array([e.step for e in events])
            vals = np.array([e.value for e in events])
            ax.plot(steps, vals, label="Episode length", color="tab:green", linestyle="--")

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --------------------------------------------
    # Optional: plot other tags
    # --------------------------------------------
    all_known_tags = total_reward_tags + step_tags + episode_tags + ep_len_tags
    other_tags = [t for t in scalar_tags if t not in all_known_tags]

    if other_tags:
        print(f"\n[INFO] Plotting {min(len(other_tags), max_extra_plots)} other tags...")
        fig2, axes2 = plt.subplots(
            int(np.ceil(min(len(other_tags), max_extra_plots) / 2)), 2, figsize=(14, 10)
        )
        axes2 = axes2.flatten()
        for i, tag in enumerate(other_tags[:max_extra_plots]):
            ax = axes2[i]
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            vals = np.array([e.value for e in events])
            ax.plot(steps, vals)
            ax.set_title(tag)
            ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    #event_file = r"logs\rsl_rl\unitree_go2_flat\2025-11-02_09-31-05_run25_mt\events.out.tfevents.1762072273.ASUSROG16.32284.0"
    #event_file = r"logs\rsl_rl\unitree_go2_flat\2025-11-17_09-57-10_run37_mt_moe_ti\events.out.tfevents.1763369838.ASUSROG16.34660.0"
    event_file = r"logs\rsl_rl\unitree_go2_flat\2025-11-17_10-46-05_run37_mt_ppo_ti\events.out.tfevents.1763372777.ASUSROG16.28100.0"
    analyze_tfevents_multitask(event_file)
