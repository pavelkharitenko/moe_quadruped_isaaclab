
"""
Multi-Task Encoder Training for Legged Robots
This script trains a task inference model using a Bayesian Nonparametric approach
to learn latent task representations that can be used by a universal policy.
"""

# 0 Imports and Inits

from legged_gym import LEGGED_GYM_ROOT_DIR
import os, sys, torch, json, datetime, matplotlib
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend

# Handle NumPy compatibility for older versions
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# import legged_gym tasks and helper functions
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.scripts.universal_policy_config import UniversalpolicyCfg

# Import project-specific modules
from MELTS.tigr.task_inference.prediction_networks import DecoderMDP
from MELTS.tigr.task_inference.dpmm_bnp import BNPModel
from MELTS.tigr.task_inference.dpmm_inference import DecoupledEncoder
from MELTS.tigr.trainer.dpmm_trainer import AugmentedTrainer
from universal_policy_utils import MinimalReplayBuffer, LatentInjectedEnvWrapper, class_to_dict, TrajectoryCollector


# main method to train DPMM-VAE and PPO

def train_multi_task_encoder(args):
    """
    Main training function for the multi-task encoder.
    
    This function:
    1. Sets up the environment and policy
    2. Initializes the task inference models
    3. Collects trajectories
    4. Trains the encoder and decoder
    5. Updates the policy with the learned task representations
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    universal_cfg = UniversalpolicyCfg()
    
    # Create timestamped log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    z_log_dir = os.path.join("z_logs", timestamp)
    os.makedirs(z_log_dir, exist_ok=True)
    
    # Save configuration to file
    config_dict = class_to_dict(UniversalpolicyCfg)
    config_path = os.path.join(z_log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Initialize Environment and Policy
    env, _ = task_registry.make_env(name="go2_universal_melts", args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name="go2_universal", args=args)
    policy = ppo_runner.get_inference_policy(device=env.device)
    task_ids = env.task_ids

    # Initialize trajectory collector
    collector = TrajectoryCollector(traj_length=universal_cfg.TRAJECTORY_LENGTH, buffer_size=universal_cfg.MAX_BUFFER_SIZE)

    # Set up logging directory
    absolute_log_dir = os.path.abspath(f"./dpmm_logs_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Initialize Bayesian Nonparametric Model
    bnp_model = BNPModel(
        save_dir=absolute_log_dir,
        gamma0=universal_cfg.bnp_model.gamma0,
        num_lap=universal_cfg.bnp_model.num_lap,
        fit_interval=universal_cfg.bnp_model.fit_interval,
        birth_kwargs=dict(
            b_startLap=universal_cfg.bnp_model.birth.startLap,
            b_stopLap=universal_cfg.bnp_model.birth.stopLap,
            b_Kfresh=universal_cfg.bnp_model.birth.Kfresh,
            b_minNumAtomsForNewComp=universal_cfg.bnp_model.birth.minNumAtomsForNewComp,
            b_minNumAtomsForTargetComp=universal_cfg.bnp_model.birth.minNumAtomsForTargetComp,
            b_minNumAtomsForRetainComp=universal_cfg.bnp_model.birth.minNumAtomsForRetainComp,
            b_minPercChangeInNumAtomsToReactivate=universal_cfg.bnp_model.birth.minPercChangeInNumAtomsToReactivate,
            b_debugOutputDir=universal_cfg.bnp_model.birth.debugOutputDir,
            b_debugWriteHTML=universal_cfg.bnp_model.birth.debugWriteHTML,
        ),
        merge_kwargs=dict(
            m_startLap=universal_cfg.bnp_model.merge.startLap,
            m_maxNumPairsContainingComp=universal_cfg.bnp_model.merge.maxNumPairsContainingComp,
            m_nLapToReactivate=universal_cfg.bnp_model.merge.nLapToReactivate,
            m_pair_ranking_procedure=universal_cfg.bnp_model.merge.pair_ranking_procedure,
            m_pair_ranking_direction=universal_cfg.bnp_model.merge.pair_ranking_direction,
        )
    )

        # Calculate shared dimension for encoder input
    
    shared_dim = (
        universal_cfg.STATE_DIM + universal_cfg.ACTION_DIM + 
        universal_cfg.REWARD_DIM + universal_cfg.STATE_DIM + 
        universal_cfg.TASKS_NUM  # using 9-class one-hot for subtype
    )
    
    # Initialize encoder and decoder models
    encoder = DecoupledEncoder(
        shared_dim, 
        universal_cfg.TIME_STEPS * shared_dim, 
        universal_cfg.Z_DIM, 
        universal_cfg.TASKS_NUM, 
        universal_cfg.TIME_STEPS, 
        'trajectory', 
        'multiplication', 
        'gru', 
        bnp_model
    )
    
    decoder = DecoderMDP(
        universal_cfg.ACTION_DIM, 
        universal_cfg.STATE_DIM, 
        universal_cfg.REWARD_DIM, 
        universal_cfg.Z_DIM, 
        2,
        universal_cfg.STATE_DIM
    )

    # Move models to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)

    # may need additional moving of decoder forward function to device here

    # 1 DATA COLLECTION: Select env for each task, collect trajs, store trajs. in replay buffer 
    
    # Collect initial trajectories for training, select n trajs per task on their respective envs and reset them
    num_tasks = universal_cfg.TASKS_NUM
    selected_envs = [] # TODO consider making one method for this and main loop beginning
    
    # for each task, find corresponding env and select first n of them
    if num_tasks == 9:
        # Base tasks 0-3
        for i in range(4):
            env_ids = torch.where(task_ids == i)[0][:universal_cfg.init_traj_collect_per_task]
            selected_envs.append(env_ids)

        # Task 4: terrain subtypes
        for terrain_type in range(5):
            matching = torch.where((task_ids == 4) & (env.terrain_types == terrain_type))[0]

            # check: only include terrain task if it has at least n running envs
            if len(matching) >= universal_cfg.trajs_collect_per_subterrain_init:
                selected_envs.append(matching[:universal_cfg.trajs_collect_per_subterrain_init])
            else:
                print(f"[INIT] Skipping terrain subtype {terrain_type} due to insufficient envs")

    elif num_tasks == 2:
        # For 2-task configuration
        for i in range(2):
            env_ids = torch.where(task_ids == i)[0][:universal_cfg.init_traj_collect_per_task]
            selected_envs.append(env_ids)

    # reset and rollout all selected environments
    selected_envs = torch.cat(selected_envs).to(env.device)
    env.reset_idx(selected_envs)
    trajs = collector.collect(env, policy, selected_envs, task_ids)

    # Initialize replay buffer with collected trajectories
    replay_buffer = MinimalReplayBuffer(trajs)

    # 2 TASK INFERENCE: Sample batch of trajs, encode to z, train encoder/decoder and DPMM to cluster tasks
    # Initialize trainer
    trainer = AugmentedTrainer(
        encoder=encoder,
        decoder=decoder,
        replay_buffer=replay_buffer,
        replay_buffer_augmented=None,
        batch_size=UniversalpolicyCfg.trainer.batch_size,
        num_classes=universal_cfg.TASKS_NUM,
        latent_dim=universal_cfg.Z_DIM,
        timesteps=universal_cfg.TIME_STEPS,
        lr_decoder=UniversalpolicyCfg.trainer.lr_decoder,
        lr_encoder=UniversalpolicyCfg.trainer.lr_encoder,
        alpha_kl_z=UniversalpolicyCfg.trainer.alpha_kl_z,  # use warm up beta
        beta_euclid=UniversalpolicyCfg.trainer.beta_euclid,
        gamma_sparsity=UniversalpolicyCfg.trainer.gamma_sparsity,
        regularization_lambda=UniversalpolicyCfg.trainer.regularization_lambda,
        use_state_diff=UniversalpolicyCfg.trainer.use_state_diff,
        state_reconstruction_clip=universal_cfg.STATE_DIM,
        use_data_normalization=UniversalpolicyCfg.trainer.use_data_normalization,
        train_val_percent=UniversalpolicyCfg.trainer.train_val_percent,
        eval_interval=UniversalpolicyCfg.trainer.eval_interval,
        early_stopping_threshold=UniversalpolicyCfg.trainer.early_stopping_threshold,
        experiment_log_dir="z_logs",
        use_regularization_loss=True,
        use_PCGrad=False,
        PCGrad_option='true_task',
        optimizer_class=torch.optim.Adam,
        log_dir=z_log_dir
    )

    # TODO could this be problematic for later epochs, e.g. this is set to 100 and we train 4000?
    # Set up warm-up schedule for KL divergence term
    beta_final = universal_cfg.warmup.beta_final   # Final KL scale
    warmup_epochs = universal_cfg.warmup.warmup_epochs  # Warm-up period
    
    # Wrap environment for latent injection: modify obs dim so latent z can be added to policy
    wrapped_env = LatentInjectedEnvWrapper(env, z_vector=None)
    ppo_runner.env = wrapped_env

    # Main training loop
    for epoch in range(universal_cfg.max_epoch):
        print(f"\n=== Epoch {epoch} ===")
        
        # Select n environments of each tasks for this epoch
        selected_envs = []
        if num_tasks == 9:

            for i in range(4):  # tasks 0-3
                selected_envs.append(torch.where(task_ids == i)[0][:universal_cfg.traj_collect_per_task])

            for terrain_type in range(5):
                matching = torch.where((task_ids == 4) & (env.terrain_types == terrain_type))[0]

                # check: only include terrain task if it has at least n running envs
                if len(matching) >= universal_cfg.traj_collect_per_subterrain:
                    selected_envs.append(matching[:universal_cfg.traj_collect_per_subterrain])
        # the 2 task case
        elif num_tasks == 2:
            for i in range(2):  # tasks 0-1
                selected_envs.append(torch.where(task_ids == i)[0][:universal_cfg.traj_collect_per_task])

        # reset and rollout all selected environments
        selected_envs = torch.cat(selected_envs).to(env.device)
        env.reset_idx(selected_envs)
        trajs = collector.collect(env, policy, selected_envs, task_ids)

        # update replay buffer
        replay_buffer.buffer.extend(trajs) 
        if len(replay_buffer.buffer) > UniversalpolicyCfg.MAX_BUFFER_SIZE:
            replay_buffer.buffer = replay_buffer.buffer[-UniversalpolicyCfg.MAX_BUFFER_SIZE:]


        # Update warm-up beta for KL divergence # TODO again check if schedule problematic
        current_beta = min(beta_final, beta_final * epoch / warmup_epochs)
        trainer.alpha_kl_z = current_beta

        # Train the encoder and decoder
        trainer.train(mixture_steps=UniversalpolicyCfg().trainer.mixture_steps, current_epoch=epoch)
        
        # Sample data for encoder input # TODO check if this is correct way to sample a batch for the encoder
        buffer_size = len(replay_buffer.buffer)
        sampled_indices = np.random.choice(buffer_size, size=UniversalpolicyCfg.trainer.batch_size, replace=False)
        e_data, _ = replay_buffer.sample_random_few_step_batch(sampled_indices, UniversalpolicyCfg.trainer.batch_size)

        # Prepare encoder input and get latent representations
        encoder_input = replay_buffer.make_encoder_data(e_data, UniversalpolicyCfg.trainer.batch_size).to(env.device)
        encoder_input = encoder_input.to(device)

        # Run encoder
        z_all, _ = encoder(encoder_input, return_probabilities=True)
        z_all = z_all.to(device)

        # Organize latent vectors by task
        z_task_map = [step[0]["task_id"] for step in trajs]
        z_by_task = [[] for _ in range(universal_cfg.TASKS_NUM)]

        for z_i, task_id in enumerate(z_task_map):
            z_by_task[task_id].append(z_all[z_i])

        # Broadcast latent vectors to environments # NOTE: Cant we really not encode all environments to z first, and then pass them to policy?
        z_broadcast = torch.zeros(env.num_envs, universal_cfg.Z_DIM, device=env.device)
        for task_id in range(universal_cfg.TASKS_NUM):
            z_list = z_by_task[task_id]
            env_indices = torch.where(task_ids == task_id)[0]

            for i, env_id in enumerate(env_indices):
                z_broadcast[env_id] = z_list[i % len(z_list)]
                
        # Save latent vectors periodically for visualization
        if epoch % 200 == 0:
            z_save_dir = os.path.join(z_log_dir, "z_checkpoints")
            os.makedirs(z_save_dir, exist_ok=True)

            z_numpy = z_all.detach().cpu().numpy()
            terrain_types = env.terrain_types.cpu().numpy()
            
            # Extract task labels
            task_labels = np.array([
                traj[0][0]["base_task"]  # Get base task from first timestep
                for traj in e_data["true_tasks"]
            ])

            # Save to file
            save_path = os.path.join(z_save_dir, f"epoch_{epoch:04d}.npz")
            np.savez_compressed(save_path, z=z_numpy, labels=task_labels)
            print(f"[INFO] Saved z latent vectors and task labels at epoch {epoch} to {save_path}")


        # 3 POLICY UPDATE: Inject latent z into policy's observation, train ppo
        # Update environment with new latent vectors and train policy
        ppo_runner.env.z_vector = z_broadcast
        ppo_runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
        ppo_runner.current_learning_iteration += 1

if __name__ == "__main__":
    args = get_args()
    train_multi_task_encoder(args)

