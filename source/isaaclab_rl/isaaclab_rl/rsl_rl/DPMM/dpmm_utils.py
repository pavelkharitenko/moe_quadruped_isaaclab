"""
Contains helper functions and classes for universal_policy
"""

import numpy as np
import torch
from collections import deque
#from legged_gym.scripts.universal_policy_config import UniversalpolicyCfg

from dataclasses import dataclass


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    task_id: int | None


class DPMMReplayBuffer:
    """
    A simplified replay buffer for storing and sampling trajectories.
    
    This buffer stores trajectories as lists of dictionaries, where each
    dictionary contains observation, action, reward, next observation, and task info.
    """

    def __init__(self, size: int, nw, device="cpu"):
        self.size = size
        self.device = device
        self.buffer = deque(maxlen=size)
        self.nw = nw  # context length

    @torch.no_grad()
    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        task_id=None,
        task_name=None,
    ):
        self.buffer.append(
            Transition(
                obs=obs.detach().to(self.device),
                action=action.detach().to(self.device),
                reward=reward.detach().to(self.device),
                next_obs=next_obs.detach().to(self.device),
                done=done.detach().to(self.device),
                task_id=task_id,
            ))

    @torch.no_grad()
    def add_batch(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        infos=None,
    ):
        num_envs = obs.shape[0]

        for i in range(num_envs):
            task_id = None
            if infos is not None:
                info_i = infos[i]
                task_id = info_i.get("task_id", None)

            self.add(
                obs=obs[i],
                action=actions[i],
                reward=rewards[i],
                next_obs=next_obs[i],
                done=dones[i],
                task_id=task_id,
            )

    def __len__(self):
        return len(self.buffer)

    def sample_indices_priority(self, batch_size):
        T = len(self.buffer)
        weights = torch.arange(1, T + 1, dtype=torch.float)
        probs = weights / weights.sum()
        indices = torch.multinomial(probs, batch_size, replacement=True)

        #print("batch_size", batch_size)
        #print("indices", indices)

        return indices.tolist()

    def sample_contexts(self, batch_size, nw):
        """
        Sample (batch_size)-amount of contexts C_t according to paper's sampling strategy Sc,
        where C_t = (c_t-nw, ... , c_t-1, c_t) and each c_t = (s, a, r, s')t.
        """
        if not (len(self.buffer) >= nw):
            return None

        indices = self.sample_indices_priority(batch_size)
        contexts = []

        for t in indices:
            start = max(0, t - nw + 1)
            window = list(self.buffer)[start:t + 1]

            contexts.append(window)

        return contexts

    def sample_random_few_step_batch(
        self,
        indices,
        batch_size,
        normalize=False,
        prio=None,
        return_sac_data=False,
    ):
        """
        API-compatible replacement for StackedReplayBuffer.sample_random_few_step_batch.

        Returns:
            e_data: dict with shape [B, T, D]
            d_data: same structure, last timestep used by decoder
        """

        # Sample contexts using your priority scheme
        contexts = self.sample_contexts(batch_size, self.nw)
        assert contexts is not None, "Not enough data to sample contexts"

        B = len(contexts)
        T = self.nw

        # Allocate numpy arrays (encoder expects numpy here)
        obs = []
        actions = []
        rewards = []
        next_obs = []
        terminals = []
        task_ids = []

        for ctx in contexts:
            # Pad context from the left if needed
            pad_len = T - len(ctx)
            if pad_len > 0:
                pad = [ctx[0]] * pad_len
                ctx = pad + ctx

            obs.append([c.obs.cpu().numpy() for c in ctx])
            actions.append([c.action.cpu().numpy() for c in ctx])
            rewards.append([c.reward.cpu().numpy() for c in ctx])
            next_obs.append([c.next_obs.cpu().numpy() for c in ctx])
            terminals.append([c.done.cpu().numpy() for c in ctx])
            # Match original true_task format: dict with base_task
            task_ids.append([c.task_id.cpu().numpy() for c in ctx])  # TODO remove task_id later

        # Convert to arrays
        e_data = dict(
            observations=np.asarray(obs, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_observations=np.asarray(next_obs, dtype=np.float32),
            terminals=np.asarray(terminals, dtype=np.uint8),
            task_ids=np.asarray(task_ids, dtype=np.float32),  # TODO remove task_id later
        )

        # Decoder only uses the last step
        d_data = e_data

        return e_data, d_data

    def make_encoder_data(self, data, batch_size, encoding_mode="trajectory", permute_samples=False):
        """
        Matches StackedReplayBuffer.make_encoder_data behavior.

        Input:
            data: dict from sample_random_few_step_batch
        Output:
            Tensor ready for encoder
        """

        observations = torch.from_numpy(data["observations"]).float()
        actions = torch.from_numpy(data["actions"]).float()
        rewards = torch.from_numpy(data["rewards"]).float()
        next_observations = torch.from_numpy(data["next_observations"]).float()
        task_ids = torch.from_numpy(data["task_ids"]).float()  # TODO remove task_id later

        # Drop last timestep (same as original)
        obs_enc = observations.detach().clone()[:, :-1, :]
        act_enc = actions.detach().clone()[:, :-1, :]
        rew_enc = rewards.detach().clone()[:, :-1, :]
        next_obs_enc = next_observations.detach().clone()[:, :-1, :]
        task_ids_enc = task_ids.detach().clone()[:, :-1, :]

        encoder_input = torch.cat(
            [obs_enc, act_enc, rew_enc, next_obs_enc, task_ids_enc],  # TODO remove task_id later
            dim=-1,
        )

        if permute_samples:
            perm = torch.randperm(encoder_input.shape[1])
            encoder_input = encoder_input[:, perm]

        if encoding_mode == "trajectory":
            encoder_input = encoder_input.view(batch_size, -1)

        return encoder_input.to(self.device)

    def get_train_val_indices(self, train_val_percent):
        """
        Returns train/val indices over valid buffer positions.
        Compatible with encoder training loops.
        """

        total = len(self.buffer)
        indices = np.arange(total)

        np.random.shuffle(indices)
        split = int(total * train_val_percent)

        train_indices = indices[:split]
        val_indices = indices[split:]

        return train_indices, val_indices


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


class TrajectoryCollector1:
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
