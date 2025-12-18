"""
Contains helper functions and classes for universal_policy
"""

import numpy as np
import torch
from collections import deque
from legged_gym.scripts.universal_policy_config import UniversalpolicyCfg


class DPMMReplayBuffer:
    """
    A simplified replay buffer for storing and sampling trajectories.
    
    This buffer stores trajectories as lists of dictionaries, where each
    dictionary contains observation, action, reward, next observation, and task info.
    """

    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)
    
    def sample_contexts(self, batch_size, nw, priority=True):
        """
        Sample (batch_size)-amount of contexts C_t according to paper's sampling strategy Sc,
        where C_t = (c_t-nw, ... , c_t-1, c_t) and each c_t = (s, a, r, s')t.
        """
        T = len(self.buffer)

        # 1) Sample start index t:

        if priority:    # create T weights: (w1,...,wT) to sample context_t (eq. 31):
            weights = np.arange(1, T + 1, dtype=np.float64)
            weights /= T*(T+1) # divide by sum of probs
            t_context = np.random.choice(T, size=batch_size, p=weights)
        else: 
            t_context = np.random.randint(0, T, size=batch_size)

        # 2) Create contexts C_t:
        contexts = []
        for t in t_context:
            start = max(0, t - nw + 1) # make sure context starts not at negative transition index
            contexts.append(list(self.buffer)[start:(t + 1)])

        return contexts



    

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


def sample_envs_per_task(env, task_ids, moe_cfg, balanced=True, debug=True):
    """
    Returns a tensor of selected environment indices for data collection.

    Args:
        env: the training environment
        task_ids: tensor of task IDs per env [num_envs]
        moe_cfg: config with traj_collect_per_task, traj_collect_per_subterrain, traj_collect_total, etc.
        balanced: if True, samples balanced per task, else global sampling.
        debug: if True, prints out distribution of sampled envs per task/subtask.
    """
    selected_envs = []

    if balanced:
        # --- Balanced sampling ---
        # Base tasks 0–2
        for i in range(3):
            matching = torch.where(task_ids == i)[0]
            env_ids = np.random.choice(matching.cpu().numpy(), size=moe_cfg.traj_collect_per_task, replace=False)
            selected_envs.append(torch.tensor(env_ids, device=env.device))

        # Terrain subtasks (task_id == 3, 5 types)
        for terrain_type in range(5):
            matching = torch.where((task_ids == 3) & (env.terrain_types == terrain_type))[0]
            if len(matching) >= moe_cfg.traj_collect_per_subterrain:
                env_ids = np.random.choice(matching.cpu().numpy(),
                                           size=moe_cfg.traj_collect_per_subterrain,
                                           replace=False)
                selected_envs.append(torch.tensor(env_ids, device=env.device))
            else:
                if debug:
                    print(f"[DEBUG] Skipping terrain subtype {terrain_type}, only {len(matching)} available")

        selected_envs = torch.cat(selected_envs).to(env.device)

    else:
        # --- Global sampling ---
        all_envs = torch.arange(env.num_envs, device=env.device)
        env_ids = np.random.choice(
            all_envs.cpu().numpy(),
            size=moe_cfg.traj_collect_total,  # total samples per epoch
            replace=False)
        selected_envs = torch.tensor(env_ids, device=env.device)

    # Debug print: show distribution
    if debug:
        task_counts = {}
        for idx in selected_envs.cpu().numpy():
            tid = int(task_ids[idx].cpu().item())
            ttype = int(env.terrain_types[idx].cpu().item()) if tid == 3 else -1
            key = (tid, ttype)
            task_counts[key] = task_counts.get(key, 0) + 1
        print(f"[DEBUG] Sampled env distribution: {task_counts}")

    return selected_envs


def select_envs_all(task_ids, terrain_types, cfg, device, debug=True):
    """
    Select *all available* envs for each task, including proper handling of subterrains.
    If cfg.traj_collect_per_task / traj_collect_per_subterrain is set, it caps per task/subtask.
    """
    num_tasks = cfg.TASKS_NUM
    selected_envs = []
    task_distribution = {}

    for task_id in range(num_tasks):
        if task_id == 3:  # terrain task, split by subtypes
            total_for_task = 0
            for subtype in range(5):
                matching = torch.where((task_ids == 3) & (terrain_types == subtype))[0]
                if cfg.traj_collect_per_subterrain > 0:
                    count = min(len(matching), cfg.traj_collect_per_subterrain)
                else:
                    count = len(matching)

                if count > 0:
                    chosen = matching[:count]
                    selected_envs.append(chosen)
                    total_for_task += len(chosen)
            task_distribution[f"task{task_id}_subtypes"] = total_for_task
        else:
            matching = torch.where(task_ids == task_id)[0]
            if cfg.traj_collect_per_task > 0:
                count = min(len(matching), cfg.traj_collect_per_task)
            else:
                count = len(matching)

            if count > 0:
                chosen = matching[:count]
                selected_envs.append(chosen)
                task_distribution[f"task{task_id}"] = len(chosen)
            else:
                task_distribution[f"task{task_id}"] = 0

    selected_envs = torch.cat(selected_envs).to(device)

    if debug:
        print("[DEBUG] Env selection distribution:")
        for k, v in task_distribution.items():
            print(f"  {k}: {v} envs")

    return selected_envs
