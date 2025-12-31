"""
Configuration for the DPMM-VAE encoder.
"""




class DpmmVaeCfg:
    log_interval = 10

    num_envs_per_iter = 4
    dpmm_buffer_size = 10_000
    context_length = 24
    max_traj_len = 1000
    batch_size = 20
    

