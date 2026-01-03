"""
Test script for Pyro-based DPMM (BNPModel)

Purpose:
- Sanity-check clustering behavior
- Visualize cluster evolution over iterations
- Verify birth / merge heuristics qualitatively

This script generates synthetic 2D data and
fits the DPMM incrementally, plotting assignments.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Fake data generation
# --------------------------------------------------

def make_sample_data(
    n_per_cluster=200,
    centers=(( -4.0, 0.0), (0.0, 0.0), (4.0, 0.0)),
    std=0.5,
    device=torch.device("cpu"),
):
    """
    Generate simple Gaussian clusters in latent space.
    """
    zs = []
    labels = []

    for k, c in enumerate(centers):
        z = torch.randn(n_per_cluster, 2, device=device) * std
        z = z + torch.tensor(c, device=device)
        zs.append(z)
        labels.append(torch.full((n_per_cluster,), k, device=device))

    z = torch.cat(zs, dim=0)
    labels = torch.cat(labels, dim=0)
    return z, labels


# --------------------------------------------------
# Plotting utilities
# --------------------------------------------------

def plot_assignments(z, Z, comp_mu=None, title="", ax=None):
    """
    Scatter plot of latent points colored by inferred cluster.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    z = z.detach().cpu()
    Z = Z.detach().cpu()

    ax.scatter(
        z[:, 0],
        z[:, 1],
        c=Z,
        cmap="tab10",
        s=12,
        alpha=0.7,
    )

    if comp_mu is not None:
        mu = comp_mu.detach().cpu()
        ax.scatter(
            mu[:, 0],
            mu[:, 1],
            c="black",
            s=120,
            marker="x",
            linewidths=2,
            label="component means",
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)


# --------------------------------------------------
# Incremental DPMM test loop
# --------------------------------------------------

def test_dpmm_evolution(
    dpmm,
    z,
    steps_per_update=200,
    num_updates=5,
    num_laps=20,
    svi_steps=500,
):
    """
    Incrementally fit the DPMM and plot clustering results.
    Mimics Algorithm 1 (buffer grows over time).
    """

    fig, axes = plt.subplots(
        1, num_updates, figsize=(4 * num_updates, 4)
    )

    if num_updates == 1:
        axes = [axes]

    for i in range(num_updates):
        # progressively larger buffer
        z_subset = z[: (i + 1) * steps_per_update]

        print(
            f"\n[DPMM TEST] update {i} | "
            f"num_points={len(z_subset)}"
        )

        # fit dpmm
        dpmm.fit(z_subset, num_laps=num_laps,svi_steps_per_lap=svi_steps)

        # cluster assignments
        resp, Z = dpmm.cluster_assignments(z_subset)

        print(f"  inferred K = {dpmm.K}")

        plot_assignments(
            z_subset,
            Z,
            comp_mu=dpmm.comp_mu,
            title=f"iter={i}, K={dpmm.K}",
            ax=axes[i],
        )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Responsibility diagnostics
# --------------------------------------------------

def plot_responsibilities(resp, max_points=200):
    """
    Visualize soft cluster assignments.
    """
    r = resp[:max_points].detach().cpu().T

    plt.figure(figsize=(6, 4))
    plt.imshow(r, aspect="auto", interpolation="nearest")
    plt.colorbar(label="responsibility")
    plt.xlabel("data index")
    plt.ylabel("cluster")
    plt.yticks(range(resp.shape[1]))
    plt.title("Responsibilities (soft assignments)")
    plt.show()


# --------------------------------------------------
# Main entry
# --------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cpu")

    # -----------------------------
    # Generate synthetic latent data
    # -----------------------------
    z, true_labels = make_sample_data(
        n_per_cluster=200,
        centers=[(-4, 0), (0, 0), (4, 0)],
        std=0.6,
        device=device,
    )

    # shuffle (important!)
    perm = torch.randperm(len(z))
    z = z[perm]
    true_labels = true_labels[perm]

    # -----------------------------
    # Initialize DPMM
    # -----------------------------
    import sys
    from pathlib import Path

    THIS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(THIS_DIR))

    from dpmm_pyro import PyroBNPModel

    dpmm = PyroBNPModel(
        latent_dim=2,
        gamma0=5.0,    # concentration parameter of DP
        K_init=1,      # start with assuming 1 cluster initially
        device=device,
    )

    # -----------------------------
    # Run evolution test
    # -----------------------------
    test_dpmm_evolution(
        dpmm,
        z,
        steps_per_update=100,
        num_updates=3,
        num_laps=10,
        svi_steps=50,
    )

    # -----------------------------
    # Final responsibility plot
    # -----------------------------
    resp, Z = dpmm.cluster_assignments(z)
    plot_responsibilities(resp)
