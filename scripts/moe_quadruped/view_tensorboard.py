from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import os




def analyze_tfevents(event_file: str):
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
        "Metrics/base_velocity/error_vel_xy",
        "Metrics/base_velocity/error_vel_yaw",
        "Episode_Reward/lin_vel_z_l2",
        "Episode_Reward/ang_vel_xy_l2"
    ]

    policy_tags = [
        "Loss/value_function",
        "Loss/surrogate",
        "Loss/entropy",
        "Policy/mean_noise_std"
    ]

    timestep_tags = [
        "Perf/collection time",
        "Perf/learning_time"
    ]

    # ---------------------
    # Create 2x2 subplot
    # ---------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Rewards subplot
    ax = axes[0]
    for tag in reward_tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            ax.plot(steps, values, label=f"{tag} (mean)")
            print(f"[INFO] {tag} mean reward stats: mean={values.mean():.2f}, max={values.max():.2f}, final={values[-1]:.2f}")

    if reward_max_tag in available:
        events = ea.Scalars(reward_max_tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        ax.plot(steps, values, "--", label=f"{reward_max_tag} (proxy max)")

    ax.set_ylabel("Reward")
    ax.set_title("Rewards")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Tracking errors subplot
    ax = axes[1]
    for tag in tracking_tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            ax.plot(steps, values, label=tag)
    ax.set_ylabel("Error / Penalty")
    ax.set_title("Tracking Errors & Penalties")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Policy & value losses subplot (normalized)
    ax = axes[2]
    for tag in policy_tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            # Normalize to [0,1]
            if values.max() > values.min():
                values = (values - values.min()) / (values.max() - values.min())
            ax.plot(steps, values, label=tag)
    ax.set_ylabel("Normalized Loss / Policy metric")
    ax.set_title("Policy & Value Losses (normalized)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Timesteps / Performance subplot
    ax = axes[3]
    for tag in timestep_tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            ax.plot(steps, values, label=tag)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Time / FPS")
    ax.set_title("Timesteps / Performance")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    #experiment_name = os.path.basename(os.path.dirname(os.path.dirname(event_file)))
    #experiment_name = "UnitreeGo2LegStand"
    experiment_name = "Unitreego2GoalTracking"
    fig.suptitle(f"Training Analysis: {experiment_name}", fontsize=11)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # replace with your actual tfevents path
    # Current flat policy
    #event_file = r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\2025-09-08_07-45-14_run03\events.out.tfevents.1757310325.ASUSROG16.35732.0"
    # Current legstand policy
    #event_file = r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\2025-09-15_10-08-13_run13_legstand\events.out.tfevents.1757930909.lrz-server1.116845.0"
    
    # Current goaltracking policy
    event_file = r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_flat\2025-09-15_08-37-53_run13\events.out.tfevents.1757925491.lrz-server1.101913.0"

    # multitask training mt4 with two envs
    event_file = r"logs\rsl_rl\unitree_go2_flat\2025-10-30_11-23-05_run21_mt32\events.out.tfevents.1761819790.ASUSROG16.38932.0"
    analyze_tfevents(event_file)
