from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import os


def analyze_tfevents(event_file: str, ignore_last_k: int = 0):
    if not os.path.exists(event_file):
        raise FileNotFoundError(f"No such file: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    available = ea.Tags()["scalars"]
    print(f"Available tags:\n{available}\n")

    # ---------------------
    # Define tags for each subplot
    # ---------------------
    reward_tags = ["Train/mean_reward"]
    reward_max_tag = "Episode_Reward/track_lin_vel_xy_exp"  # example max proxy

    tracking_tags = [
        "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw", "Episode_Reward/lin_vel_z_l2",
        "Episode_Reward/ang_vel_xy_l2"
    ]

    policy_tags = ["Loss/value_function", "Loss/surrogate", "Loss/entropy", "Policy/mean_noise_std"]

    timestep_tags = ["Perf/collection time", "Perf/learning_time"]

    # ---------------------
    # Create 2x2 subplot
    # ---------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    def trim(steps, values):
        """Helper to drop last k iterations."""
        if ignore_last_k > 0 and len(values) > ignore_last_k:
            return steps[:-ignore_last_k], values[:-ignore_last_k]
        return steps, values

    # --- Rewards subplot
    ax = axes[0]
    for tag in reward_tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])

            # --- APPLY IGNORE LAST K ---
            steps, values = trim(steps, values)

            ax.plot(steps, values, label=f"{tag} (mean)")

            print(f"[INFO] {tag} mean reward stats: "
                  f"mean={values.mean():.2f}, max={values.max():.2f}, final={values[-1]:.2f}")

    if reward_max_tag in available:
        events = ea.Scalars(reward_max_tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        steps, values = trim(steps, values)
        ax.plot(steps, values, "--", label=f"{reward_max_tag} (proxy max)")

    ax.set_ylabel("Reward")
    ax.set_title("Rewards")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # --- Tracking errors subplot
    ax = axes[1]
    for tag in tracking_tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            steps, values = trim(steps, values)
            ax.plot(steps, values, label=tag)
    ax.set_ylabel("Error / Penalty")
    ax.set_title("Tracking Errors & Penalties")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # --- Policy & value losses subplot
    ax = axes[2]
    for tag in policy_tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            steps, values = trim(steps, values)

            if values.max() > values.min():
                values = (values - values.min()) / (values.max() - values.min())

            ax.plot(steps, values, label=tag)
    ax.set_ylabel("Normalized Loss / Policy metric")
    ax.set_title("Policy & Value Losses (normalized)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # --- Performance subplot
    ax = axes[3]
    for tag in timestep_tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            steps, values = trim(steps, values)
            ax.plot(steps, values, label=tag)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Time / FPS")
    ax.set_title("Timesteps / Performance")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Training Analysis", fontsize=11)
    plt.tight_layout()
    plt.show()
    plt.savefig('run_41_ppo_with_onehot_vec.png')

    # ---------------------
    # Plot Mean Episode Length
    # ---------------------
    if "Train/mean_episode_length" in available:
        events = ea.Scalars("Train/mean_episode_length")
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        steps, values = trim(steps, values)

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, label="Train/mean_episode_length")
        plt.xlabel("Training steps")
        plt.ylabel("Mean Episode Length")
        plt.title("Mean Episode Length per Training Step")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        print(f"[INFO] mean_episode_length: "
              f"mean={values.mean():.2f}, max={values.max():.2f}, final={values[-1]:.2f}")
        plt.show()
        plt.savefig("mean_ep_length_flat_2000.png")


if __name__ == "__main__":
    event_file = r"logs/rsl_rl/unitree_go2_flat/2025-11-25_06-09-22_run41_mt_ppo_fl_oh/events.out.tfevents.1764050993.lrz-server1.3600078.0"

    # ignore last 200 iterations
    analyze_tfevents(event_file, ignore_last_k=0)
