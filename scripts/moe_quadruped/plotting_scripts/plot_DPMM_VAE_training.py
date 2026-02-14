import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ------------------------------------------------------------------
# Event file (your path)
# ------------------------------------------------------------------

# 3 tasks
#event_file = r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-12_17-37-33_dpmm_final_cond_3t_43\2026-02-12_17-37-33_dpmm_final_cond_3t_43\dpmm_tsne_plots\events.out.tfevents.1770907052.asus-G16.4446.0"

# 2 tasks
event_file = r"C:\Users\Pavel\IsaacLab\logs\rsl_rl\unitree_go2_multitask\2026-02-12_15-34-12_dpmm_final_cond_2t_44\2026-02-12_15-34-12_dpmm_final_cond_2t_44\dpmm_tsne_plots\events.out.tfevents.1770899652.asus-G16.5625.0"

event_acc = EventAccumulator(event_file)
event_acc.Reload()
# ------------------------------------------------------------------
# VAE / Mixture Tags
# ------------------------------------------------------------------

print("Available scalar tags:")
for t in event_acc.Tags()["scalars"]:
    print(" -", t)
vae_tags_top = {
    "ELBO Loss": "training/ti_mixture_elbo_loss",
    "Total Loss": "training/ti_mixture_loss",
}

vae_tags_bottom = {
    "KL(z)": "training/ti_mixture_klz_loss",
    "NLL": "training/ti_mixture_nll",
    "State Loss": "training/ti_mixture_state_losses",
    "Reward Loss": "training/ti_mixture_reward_losses",
    "Clustering": "training/ti_mixture_clustering_losses",
}

# ------------------------------------------------------------------
# Colors (same seaborn family you used)
# ------------------------------------------------------------------
colors = {
    "ELBO Loss": "#4C72B0",
    "Total Loss": "#55A868",
    "KL(z)": "#DD8452",
    "NLL": "#8172B2",
    "State Loss": "#64B5CD",
    "Reward Loss": "#C44E52",
    "Clustering": "#CCB974",
}

# ------------------------------------------------------------------
# Helpers (same smoothing style)
# ------------------------------------------------------------------
def running_average(x, window=40):
    if window <= 1:
        return x
    pad = window // 2
    x_padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(x_padded, kernel, mode="valid")
    return smoothed[:len(x)]


# ------------------------------------------------------------------
# Load event file
# ------------------------------------------------------------------
event_acc = EventAccumulator(event_file)
event_acc.Reload()

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
def load_tag(tag):
    events = event_acc.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    values = running_average(values)
    return steps, values


data_top = {}
data_bottom = {}
steps = None

for name, tag in vae_tags_top.items():
    s, v = load_tag(tag)
    data_top[name] = v
    steps = s

for name, tag in vae_tags_bottom.items():
    s, v = load_tag(tag)
    data_bottom[name] = v

# Align lengths
min_len = min(
    [len(v) for v in list(data_top.values()) + list(data_bottom.values())]
)
steps = steps[:min_len]

for k in data_top:
    data_top[k] = data_top[k][:min_len]

for k in data_bottom:
    data_bottom[k] = data_bottom[k][:min_len]

# ------------------------------------------------------------------
# Plot (2x1 layout — thesis style)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(4, 5), dpi=300, sharex=True)

# ==============================
# Top: ELBO + Total Loss
# ==============================
ax = axes[0]
for name in data_top:
    ax.plot(
        steps,
        data_top[name],
        linewidth=2,
        label=name,
        color=colors[name],
    )

ax.set_ylabel("Loss", fontsize=10)
ax.tick_params(axis="both", labelsize=10)
ax.grid(True)
ax.legend(fontsize=9)

# ==============================
# Bottom: Loss Components
# ==============================
ax = axes[1]
for name in data_bottom:
    ax.plot(
        steps,
        data_bottom[name],
        linewidth=2,
        label=name,
        color=colors[name],
    )

ax.set_xlabel("iteration steps", fontsize=10)
ax.set_ylabel("Loss", fontsize=10)
ax.tick_params(axis="both", labelsize=10)
ax.grid(True)
ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()
