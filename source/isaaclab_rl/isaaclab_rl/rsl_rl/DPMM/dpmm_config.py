"""
Configuration for the DPMM-VAE encoder.
"""


class DpmmVaeCfg:
    """
    Configuration for own DPMM-VAE implemted version (not using bnpy package)
    """
    log_interval = 10

    num_envs_per_iter = 4
    dpmm_buffer_size = 10_000
    context_length = 24
    max_traj_len = 1000
    batch_size = 20


class DpmmVaeCfgBnpy:
    """
    Configuration for native MELTS DPMM-VAE (using bnpy)
    """

    # General / logging
    log_interval = 10
    reselect_envs_interval = 1000
    num_tasks = 2

    # DPMM Buffer & Trajectory
    num_envs_per_iter = 0.2  # fraction percentage of total envs to sample trajectories from
    dpmm_buffer_size = 10_000
    context_length = 24
    max_traj_len = 1000
    batch_size = 20

    # Environment / model dims
    trajectory_length = 64
    time_steps = 25

    z_dim = 12

    # NOTE: legacy MELTS used this for GMM/DPMM
    #state_dim = 247  # obs_dim + z_dim in old setup

    # BNP / DPMM (bnpy)
    class bnp_model:
        gamma0 = 5.0
        num_lap = 10
        fit_interval = "adaptive"

        class birth:
            start_lap = 1
            stop_lap = 5
            k_fresh = 2
            min_num_atoms_for_new_comp = 16
            min_num_atoms_for_target_comp = 16
            min_num_atoms_for_retain_comp = 16
            min_perc_change_to_reactivate = 0.05
            debug_output_dir = None
            debug_write_html = 0

        class merge:
            start_lap = 5
            max_num_pairs_containing_comp = 50
            n_lap_to_reactivate = 2
            pair_ranking_procedure = "obsmodel_elbo"
            pair_ranking_direction = "descending"

    # Training
    class trainer:
        batch_size = 20
        batch_size_rollout = 256
        lr_decoder = 3e-4
        lr_encoder = 3e-4

        alpha_kl_z = 1e-4
        beta_euclid = 5e-4
        gamma_sparsity = 1e-3

        regularization_lambda = 0.1
        use_state_diff = False
        use_data_normalization = True

        train_val_percent = 1.0
        eval_interval = 50
        early_stopping_threshold = 500

        experiment_log_dir = "z_logs"
        log_dir = None

        use_regularization_loss = True
        use_pcgrad = False
        pcgrad_option = "true_task"
        optimizer_class = "Adam"

        mixture_steps = 128

    # Warmup
    class warmup:
        beta_final = 1e-4
        warmup_epochs = 100
