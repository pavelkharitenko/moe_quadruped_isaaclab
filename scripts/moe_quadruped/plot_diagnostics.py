"""
Compare PPO training diagnostics between single-task and multi-task runs.

Usage:
    python compare_runs.py --single path/to/single/log_dir --multi path/to/multi/log_dir
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


def load_diagnostics(log_dir):
    """Load main and per-task CSV diagnostics from a log directory."""
    files = glob(os.path.join(log_dir, "training_diagnostics*.csv"))
    data = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        data[name] = df
    return data


def summarize(df):
    """Return summary statistics for the key numeric columns."""
    stats = {}
    for col in df.columns:
        if col == "iteration":
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        stats[col] = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "final": series.iloc[-1],
        }
    return pd.DataFrame(stats).T


def compare_diagnostics(single_dir, multi_dir, save_fig=False):
    """Plot side-by-side comparison between single-task and multi-task runs."""
    single = load_diagnostics(single_dir)
    multi = load_diagnostics(multi_dir)

    single_main = single.get("training_diagnostics")
    multi_main = multi.get("training_diagnostics")

    if single_main is None or multi_main is None:
        raise FileNotFoundError("Both runs must contain a training_diagnostics.csv file.")

    metrics = [
        "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_fraction", "learning_rate",
        "mean_episode_reward", "mean_episode_length",
    ]

    shared_metrics = [m for m in metrics if m in single_main.columns and m in multi_main.columns]
    n_metrics = len(shared_metrics)

    # --- Create subplots ---
    fig, axs = plt.subplots(nrows=(n_metrics + 1) // 2, ncols=2, figsize=(12, 3.5 * ((n_metrics + 1) // 2)))
    axs = axs.flatten()

    for i, metric in enumerate(shared_metrics):
        ax = axs[i]
        ax.plot(single_main["iteration"], single_main[metric], label="Single-task", linewidth=2)
        ax.plot(multi_main["iteration"], multi_main[metric], label="Multi-task", linewidth=2, alpha=0.8)
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    if save_fig:
        out_path = os.path.join(os.getcwd(), "ppo_run_comparison.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved comparison plot → {out_path}")
    else:
        plt.show()

    # --- Print statistics ---
    print("\n=== 📊 Single-task statistics ===")
    print(summarize(single_main).round(4))
    print("\n=== 📊 Multi-task statistics ===")
    print(summarize(multi_main).round(4))

    # --- Compare final values directly ---
    print("\n=== 🔍 Final comparison (last iteration) ===")
    final_compare = []
    for m in shared_metrics:
        final_single = single_main[m].iloc[-1]
        final_multi = multi_main[m].iloc[-1]
        final_compare.append((m, final_single, final_multi, final_multi - final_single))
    comp_df = pd.DataFrame(final_compare, columns=["metric", "single_final", "multi_final", "delta(multi-single)"])
    print(comp_df.round(4))

    return comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, required=True, help="Path to single-task log_dir/")
    parser.add_argument("--multi", type=str, required=True, help="Path to multi-task log_dir/")
    parser.add_argument("--save_fig", action="store_true", help="Save comparison figure instead of showing it.")
    args = parser.parse_args()

    compare_diagnostics(args.single, args.multi, save_fig=args.save_fig)
