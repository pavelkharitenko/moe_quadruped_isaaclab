from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import itertools


def collect_scalars(ea, pattern, ignore_last_k=0):
    """Return {tag: (steps, values)} for scalar tags matching regex."""
    tags = ea.Tags()["scalars"]
    matched = [t for t in tags if re.search(pattern, t)]
    data = {}

    for tag in matched:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events], dtype=float)

        # sanitize NaNs/Infs for plotting
        values = np.nan_to_num(values, nan=np.nan, posinf=np.nan, neginf=np.nan)

        if ignore_last_k > 0 and len(values) > ignore_last_k:
            steps = steps[:-ignore_last_k]
            values = values[:-ignore_last_k]

        data[tag] = (steps, values)

    return data


def analyze_reward_terms(event_file, ignore_last_k=0, save_path="reward_analysis.png"):

    if not os.path.exists(event_file):
        raise FileNotFoundError(event_file)

    print("[INFO] Loading:", event_file)
    ea = EventAccumulator(event_file)
    ea.Reload()

    print(f"[INFO] Loaded {len(ea.Tags()['scalars'])} scalar tags.")

    # -----------------------------
    # Regex patterns
    # -----------------------------
    flatvel_pattern = r"Rewards/flatVel/Episode_Reward/.*"
    legstand_pattern = r"Rewards/legStand/Episode_Reward/.*"
    global_pattern = r"^(Train/.*|Metrics/.*|Loss/.*)"
    all_rewards_pattern = r"Episode_Reward/.*"

    # -----------------------------
    # Load data
    # -----------------------------
    flatvel_data = collect_scalars(ea, flatvel_pattern, ignore_last_k)
    legstand_data = collect_scalars(ea, legstand_pattern, ignore_last_k)
    global_data = collect_scalars(ea, global_pattern, ignore_last_k)
    all_rewards = collect_scalars(ea, all_rewards_pattern, ignore_last_k)

    # Load Train/mean_reward for overlay
    mean_reward_data = collect_scalars(ea, r"Train/mean_reward", ignore_last_k)
    mean_reward_steps, mean_reward_values = list(mean_reward_data.values())[0] if mean_reward_data else ([], [])

    # -----------------------------
    # Create 2x2 subplot
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    groups = [
        ("FlatVel Reward Terms", flatvel_data),
        ("LegStand Reward Terms", legstand_data),
        ("Global / MT Terms", global_data),
        ("All Episode Reward Terms", all_rewards),
    ]

    # Colormap for many curves
    color_cycle = plt.cm.get_cmap("tab20").colors  # 20 distinct colors
    color_iter = itertools.cycle(color_cycle)

    # -----------------------------
    # Plot each group
    # -----------------------------
    for ax, (title, group) in zip(axes, groups):
        if len(group) == 0:
            ax.set_title(title + " (no matching tags)")
            ax.axis('off')
            continue

        print(f"\n[INFO] Plotting {title}, {len(group)} tags")

        for tag, (steps, values) in group.items():
            color = next(color_iter)
            ax.plot(steps, values, label=tag, color=color)

            # Plot episode-average points where not NaN
            mask = ~np.isnan(values)
            ax.scatter(steps[mask], values[mask], marker='x', color=color, s=20, alpha=0.7)

            # Print stats
            if len(values) > 0:
                print(f"  {tag}: mean={np.nanmean(values):.4f}, "
                      f"max={np.nanmax(values):.4f}, min={np.nanmin(values):.4f}")

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=6)

    # Overlay Train/mean_reward
    #if len(mean_reward_steps) > 0:
    #    ax.plot(mean_reward_steps, mean_reward_values, 'k-', linewidth=2, label="Train/mean_reward")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\n[INFO] Saved figure to: {save_path}")


if __name__ == "__main__":
    event_file = r"logs/rsl_rl/unitree_go2_flat/2025-11-25_09-43-10_run43_mt_moe_fl_oh/events.out.tfevents.1764063822.lrz-server1.3617652.0"
    analyze_reward_terms(event_file, ignore_last_k=0, save_path="reward_term_analysis.png")
