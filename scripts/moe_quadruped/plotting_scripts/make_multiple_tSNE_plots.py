import numpy as np
import os
import json
import matplotlib.pyplot as plt


def get_tsne_folders(run_dir):
    """
    Returns sorted list of tsne subfolders containing embeddings.npz
    """
    tsne_root = os.path.join(run_dir, "tsne_logs")

    if not os.path.isdir(tsne_root):
        return []

    folders = [
        os.path.join(tsne_root, f)
        for f in os.listdir(tsne_root)
        if os.path.isdir(os.path.join(tsne_root, f))
        and os.path.exists(os.path.join(tsne_root, f, "embeddings.npz"))
    ]

    # Sort by epoch number (extracted from metadata)
    def get_epoch(folder):
        with open(os.path.join(folder, "metadata.json"), "r") as f:
            meta = json.load(f)
        return meta["epoch"]

    folders = sorted(folders, key=get_epoch)
    return folders


import math

def plot_run_tsne(run_dir):

    folders = get_tsne_folders(run_dir)
    M = len(folders)

    if M == 0:
        print(f"No tsne logs found in {run_dir}")
        return

    rows = 3
    cols = math.ceil(M / rows)

    # ----------------------------
    # Styling (unchanged)
    # ----------------------------
    task_names = ["FlatVel", "Handstand", "Legstand", "Crawl"]
    task_colors = {
        "FlatVel": "#2C7AC4",
        "Handstand": "#55A868",
        "Legstand": "#DD8452",
        "Crawl": "#8172B2",
    }

    cluster_palette = [
        "#63c1db",
        "#76B7B2",
        "#F28E2B",
        "#B07AA1",
        "#E15759",
        "#ED48E2",
        "#82A14F",
    ]

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        dpi=300,
        squeeze=False,
    )

    for idx, folder in enumerate(folders):

        row = idx % rows
        col = idx // rows

        ax = axes[row, col]

        data = np.load(os.path.join(folder, "embeddings.npz"))
        z_2d = data["z_2d"]
        cluster_ids = data["cluster_ids"]
        gt_labels = data["gt_labels"]

        with open(os.path.join(folder, "metadata.json"), "r") as f:
            metadata = json.load(f)

        max_clusters = len(cluster_palette)
        cluster_colors = [
            cluster_palette[cid % max_clusters] for cid in cluster_ids
        ]

        gt_edge_colors = [
            task_colors[task_names[t]] for t in gt_labels
        ]

        # Filled markers (clusters)
        ax.scatter(
            z_2d[:, 0],
            z_2d[:, 1],
            c=cluster_colors,
            s=20,
            alpha=0.85,
            linewidths=0.125,
            edgecolors="none",
        )

        # Hollow markers (GT)
        ax.scatter(
            z_2d[:, 0],
            z_2d[:, 1],
            facecolors="none",
            edgecolors=gt_edge_colors,
            s=25,
            linewidths=0.7,
        )

        #ax.set_title(
        #    f"Epoch {metadata['epoch']} | "
        #    f"ARI: {metadata['ARI']:.2f}",
        #    fontsize=9,
        #)

        ax.set_xlabel("t-SNE dim 1", fontsize=8)
        ax.set_ylabel("t-SNE dim 2", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)

    # ----------------------------
    # Hide unused axes
    # ----------------------------
    total_axes = rows * cols
    for empty_idx in range(M, total_axes):
        row = empty_idx % rows
        col = empty_idx // rows
        axes[row, col].axis("off")

    # ----------------------------
    # Single legend
    # ----------------------------
    unique_clusters = sorted(set(cluster_ids))

    cluster_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=cluster_palette[i % len(cluster_palette)],
            markeredgecolor="none",
            label=f"Cluster {i}",
            markersize=8,
        )
        for i in unique_clusters
    ]

    task_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=color,
            label=name,
            markersize=8,
            linewidth=1.6,
        )
        for name, color in task_colors.items()
    ]

    fig.legend(
        handles=cluster_handles,
        title="DPMM clusters",
        loc="upper left",
        fontsize=9,
    )

    fig.legend(
        handles=task_handles,
        title="Ground-truth tasks",
        loc="upper right",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_multiple_runs(run_dirs):
    for run_dir in run_dirs:
        print(f"\nPlotting run: {run_dir}")
        plot_run_tsne(run_dir)


run_dirs = []

plot_multiple_runs(run_dirs)