"""
Contains helper functions and classes for DPMM data collection and training.
"""

import numpy as np
import torch
from collections import deque


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
        #task_ids = []

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
            #task_ids.append([c.task_id.cpu().numpy() for c in ctx])  # TODO remove task_id later

        # Convert to arrays
        e_data = dict(
            observations=np.asarray(obs, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_observations=np.asarray(next_obs, dtype=np.float32),
            terminals=np.asarray(terminals, dtype=np.uint8),
            #task_ids=np.asarray(task_ids, dtype=np.float32),  # TODO remove task_id later
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
        #task_ids = torch.from_numpy(data["task_ids"]).float()

        # Drop last timestep (same as original)
        obs_enc = observations.detach().clone()[:, :-1, :]
        act_enc = actions.detach().clone()[:, :-1, :]
        rew_enc = rewards.detach().clone()[:, :-1, :]
        next_obs_enc = next_observations.detach().clone()[:, :-1, :]
        #task_ids_enc = task_ids.detach().clone()[:, :-1, :]

        encoder_input = torch.cat(
            [obs_enc, act_enc, rew_enc, next_obs_enc],
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

