

def plot_saved_tsne(folder_path):

    data = np.load(os.path.join(folder_path, "embeddings.npz"))
    z_2d = data["z_2d"]
    cluster_ids = data["cluster_ids"]
    gt_labels = data["gt_labels"]

    with open(os.path.join(folder_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

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

    fig, ax = plt.subplots(figsize=(7, 5))

    max_clusters = len(cluster_palette)
    cluster_colors = [cluster_palette[cid % max_clusters] for cid in cluster_ids]

    ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=cluster_colors,
        s=80,
        alpha=0.85,
        edgecolors="none",
    )

    gt_edge_colors = [task_colors[task_names[t]] for t in gt_labels]

    ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        facecolors="none",
        edgecolors=gt_edge_colors,
        s=100,
        linewidths=1.6,
    )

    ax.set_title(
        f"Epoch {metadata['epoch']} | "
        f"Clusters: {metadata['num_clusters']} | "
        f"ARI: {metadata['ARI']:.3f} | "
        f"NMI: {metadata['NMI']:.3f}",
        fontsize=11,
    )

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()