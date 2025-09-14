"""
Contains helper functions and classes for universal_policy
"""

import numpy as np
import torch
from legged_gym.scripts.universal_policy_config import UniversalpolicyCfg


class MinimalReplayBuffer:
    """
    A simplified replay buffer for storing and sampling trajectories.
    
    This buffer stores trajectories as lists of dictionaries, where each
    dictionary contains observation, action, reward, next observation, and task info.
    """
    
    def __init__(self, buffer):
        """
        Initialize the replay buffer.
        
        Args:
            buffer: Initial list of trajectories to populate the buffer
        """
        self.buffer = buffer

    def sample_random_few_step_batch(self, indices=None, batch_size=128, normalize=False, prio=None):
        """
        Sample a batch of trajectories from the buffer.
        
        Args:
            indices: Specific indices to sample from (if None, sample from all)
            batch_size: Number of trajectories to sample
            normalize: Whether to normalize the data (not implemented)
            prio: Priority sampling weights (not implemented)
            
        Returns:
            Tuple of (data_dict, data_dict) where each data_dict contains:
                - observations: array of shape (B, T, obs_dim)
                - actions: array of shape (B, T, act_dim)
                - rewards: array of shape (B, T, 1)
                - next_observations: array of shape (B, T, obs_dim)
                - terminals: array of zeros with shape (B, T, 1)
                - true_tasks: array containing task information
        """
        # Select indices to sample
        if indices is None:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        else:
            indices = np.random.choice(indices, size=batch_size, replace=False)

        # Extract the sampled trajectories
        data = [self.buffer[i] for i in indices]

        # Get dimensions
        B = len(data)  # Batch size
        T = len(data[0])  # Trajectory length
        
        # Stack the trajectory data into arrays
        obs = np.stack([[step["obs"] for step in traj] for traj in data])
        act = np.stack([[step["act"] for step in traj] for traj in data])
        rew = np.stack([[step["rew"] for step in traj] for traj in data])
        rew = np.expand_dims(rew, axis=-1)  # Add reward dimension
        next_obs = np.stack([[step["next_obs"] for step in traj] for traj in data])
        terminals = np.zeros_like(rew)  # All zeros (no terminal states)
        
        # Extract task information
        true_tasks = np.array([[[{"base_task": step["task_subtype_id"]}] for step in traj] for traj in data])

        # Create data dictionary
        data_dict = {
            'observations': obs,
            'actions': act,
            'rewards': rew,
            'next_observations': next_obs,
            'terminals': terminals,
            'true_tasks': true_tasks,
        }
        
        return (data_dict, data_dict)  # Return tuple for compatibility

    def make_encoder_data(self, data_dict, batch_size):
        """
        Prepare data for the encoder by concatenating states, actions, rewards, 
        next states, and one-hot task encodings.
        
        Args:
            data_dict: Dictionary containing trajectory data
            batch_size: Number of trajectories in the batch
            
        Returns:
            Tensor of shape (batch_size, time_steps * feature_dim) for encoder input
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract components from data dictionary
        states = torch.tensor(data_dict["observations"], dtype=torch.float32, device=device)
        actions = torch.tensor(data_dict["actions"], dtype=torch.float32, device=device)
        rewards = torch.tensor(data_dict["rewards"], dtype=torch.float32, device=device)
        next_states = torch.tensor(data_dict["next_observations"], dtype=torch.float32, device=device)

        # Extract task information and create one-hot encodings
        true_tasks_raw = data_dict["true_tasks"]  # shape: (B, T, 1)
        task_subtype_ids = np.array([[step[0]["base_task"] for step in traj] for traj in true_tasks_raw])
        task_subtype_ids = torch.tensor(task_subtype_ids, dtype=torch.long, device=device)
        one_hot = torch.nn.functional.one_hot(task_subtype_ids, num_classes=9).float()

        # Concatenate all components along the feature dimension
        sequence = torch.cat([states, actions, rewards, next_states, one_hot], dim=-1)
        
        # Flatten the time and feature dimensions for encoder input
        B, T, D = sequence.shape
        sequence = sequence.reshape(B, T * D)

        # Verify batch size consistency
        assert B == batch_size, f"Batch size mismatch: expected {batch_size}, got {B}"
        
        return sequence

    def get_train_val_indices(self, train_val_percent):
        """
        Split the buffer indices into training and validation sets.
        
        Args:
            train_val_percent: Percentage of data to use for training (0-1)
            
        Returns:
            Tuple of (train_indices, validation_indices)
        """
        total = len(self.buffer)
        indices = np.arange(total)
        np.random.shuffle(indices)
        split = int(total * train_val_percent)
        return indices[:split], indices[split:]


class LatentInjectedEnvWrapper:
    """
    Environment wrapper that injects latent task representations into observations.
    
    This wrapper modifies the environment's observations to include a latent
    task representation vector, allowing the policy to adapt to different tasks.
    """
    
    def __init__(self, env, z_vector):
        """
        Initialize the wrapper.
        
        Args:
            env: The original environment to wrap
            z_vector: Latent task representation vector to inject
        """
        self.env = env
        self.z_vector = z_vector
        self.num_obs = UniversalpolicyCfg().OBS_DIM
        self.z_dim = UniversalpolicyCfg().Z_DIM
        self.total_obs_dim = self.num_obs + self.z_dim

    def get_observations(self):
        """
        Get observations from the environment and inject the latent task vector.
        
        Returns:
            Tuple of (modified observations, extras)
        """
        obs, extras = self.env.get_observations()
        # Inject latent vector into the last dimensions of the observation
        obs[:, -self.z_dim:] = self.z_vector.to(obs.device)
        return obs, extras

    def step(self, actions):
        """
        Take a step in the environment and modify the observations.
        
        Args:
            actions: Actions to take in the environment
            
        Returns:
            Tuple of (obs_buf, rew_buf, reset_buf, extras) with modified observations
        """
        # Step the environment
        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = self.env.step(actions)
        
        # Inject latent vector into the observations
        obs_buf[:, -self.z_dim:] = self.z_vector.to(obs_buf.device)
        
        # Return the modified outputs
        return obs_buf, rew_buf, reset_buf, extras

    def __getattr__(self, attr):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, attr)


class TrajectoryCollector:
    """
    Collects trajectories from the environment using a policy.
    
    This class handles the collection of trajectories for training the
    task inference model.
    """
    
    def __init__(self, traj_length, buffer_size):
        """
        Initialize the trajectory collector.
        
        Args:
            traj_length: Length of each trajectory to collect
            buffer_size: Maximum size of the replay buffer
        """
        self.traj_length = traj_length
        self.buffer_size = buffer_size
        self.training_buffer = []

    def collect(self, env, policy, selected_envs, task_ids):
        """
        Collect trajectories from selected environments.
        
        Args:
            env: The environment to collect trajectories from
            policy: The policy to use for action selection
            selected_envs: Indices of environments to collect from
            task_ids: Task IDs for each environment
            
        Returns:
            List of collected trajectories
        """
        # Get initial observations
        obs = env.get_observations()
        prev_obs = obs.clone()
        
        # Initialize trajectory storage
        trajs = [[] for _ in selected_envs]

        # Collect trajectories step by step
        for _ in range(self.traj_length):
            with torch.no_grad():
                act = policy(obs)  # Get actions from policy
                
            # Step the environment
            obs, _, rews, _, _ = env.step(act)
            
            # Store the transition for each selected environment
            for i, env_id in enumerate(selected_envs):
                task_id = int(task_ids[env_id].item())
                terrain_type = int(env.terrain_types[env_id].item())
                
                # Determine task subtype (base task or terrain-specific)
                task_subtype = task_id if task_id < 4 else 5 + terrain_type

                # Store the transition
                trajs[i].append({
                    "obs": prev_obs[env_id].cpu().numpy(),
                    "act": act[env_id].cpu().numpy(),
                    "rew": rews[env_id].cpu().numpy(),
                    "next_obs": obs[env_id].cpu().numpy(),
                    "task_id": task_id,
                    "task_subtype_id": task_subtype
                })
                
            prev_obs = obs.clone()  # Update previous observations
        
        # Update the training buffer
        self.training_buffer = trajs
        
        return trajs

    def prepare_encoder_input(self):
        """
        Prepare the collected trajectories for encoder input.
        
        Returns:
            Tensor of shape (num_trajectories, traj_length, feature_dim)
        """
        sequences = []
        for traj in self.training_buffer:
            # Extract components from each step in the trajectory
            states = np.stack([step["obs"][:UniversalpolicyCfg().STATE_DIM] for step in traj])
            actions = np.stack([step["act"] for step in traj])
            rewards = np.stack([step["rew"] for step in traj])
            rewards = rewards.reshape(-1, 1)  # Add dimension
            next_states = np.stack([step["next_obs"][:UniversalpolicyCfg().STATE_DIM] for step in traj])
            
            # Concatenate components
            seq = np.concatenate([states, actions, rewards, next_states], axis=1)
            sequences.append(seq)
            
        return torch.tensor(np.stack(sequences), dtype=torch.float32)


def class_to_dict(cls):
    """
    Convert a class to a dictionary representation.
    
    Args:
        cls: The class to convert
        
    Returns:
        Dictionary representation of the class attributes
    """
    d = {}
    for attr in dir(cls):
        if attr.startswith('__'):
            continue
        val = getattr(cls, attr)
        if isinstance(val, type):
            d[attr] = class_to_dict(val)
        else:
            d[attr] = val
    return d

class UniversalpolicyCfg: 
    TRAJECTORY_LENGTH = 64
    TASKS_NUM = 9
    MAX_BUFFER_SIZE = 10000
    #just temporary
    #OBS_DIM=235
    OBS_DIM=235
    ACTION_DIM = 12  # robot DOF count
    REWARD_DIM = 1
    Z_DIM = 12
    TIME_STEPS = 64
    FIT_INTERVAL = 10
    #for dpmm
    #STATE_DIM = OBS_DIM+Z_DIM  # actual environment observation size
    #for gmm
    STATE_DIM=247
    max_epoch=4000
    traj_collect_per_task=100 #how many robots' traj are collected per task per step
    traj_collect_per_subterrain=50 #how many robots' traj arec collected for each terrain
    init_traj_collect_per_task=800
    class bnp_model:
        gamma0 = 5.0
        num_lap = 10
        fit_interval = "adaptive"

        class birth:
            startLap = 1
            stopLap = 5
            Kfresh = 2
            minNumAtomsForNewComp = 16
            minNumAtomsForTargetComp = 16
            minNumAtomsForRetainComp = 16
            minPercChangeInNumAtomsToReactivate = 0.05
            debugOutputDir = None
            debugWriteHTML = 0

        class merge:
            startLap = 5
            maxNumPairsContainingComp = 50
            nLapToReactivate = 2
            pair_ranking_procedure = 'obsmodel_elbo'
            pair_ranking_direction = 'descending'

    class trainer:
        batch_size=1024
        batch_size_rollout=256
        lr_decoder = 3e-4
        lr_encoder = 3e-4
        alpha_kl_z = 0.0001
        beta_euclid = 0.0005
        gamma_sparsity = 0.001
        regularization_lambda = 0.1
        use_state_diff = False
        use_data_normalization = True
        train_val_percent = 1.0
        eval_interval = 50
        early_stopping_threshold = 500
        experiment_log_dir = "z_logs"
        use_regularization_loss = True
        use_PCGrad = False
        PCGrad_option = 'true_task'
        optimizer_class = 'Adam'
        log_dir = None
        mixture_steps=128

    class warmup:
        beta_final = 0.0001
        warmup_epochs = 100

    class ppo:
        num_learning_iterations = 1
        init_at_random_ep_len = True
    