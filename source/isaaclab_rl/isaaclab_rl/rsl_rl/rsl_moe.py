import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import os
import time
import statistics
import pandas as pd
import csv
import numpy as np
from torch.distributions import Normal

import rsl_rl
from rsl_rl.utils import store_code_state
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)


class MyActorCritic(ActorCritic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # custom AC logic here




class MoEActorCritic(ActorCritic):
    """
    ActorCritic with n expert Actor networks, and 1 Shared Critic network.
    Softmax is applied on the input, and forwarded to all experts
    """
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_experts=4,
        gating_hidden_dims=[128, 128],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        **kwargs,
    ):
        # call parent to build critic, and reuse its activation resolution
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )


        activation = resolve_nn_activation(activation)


        self.num_experts = num_experts

        # create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_actor_obs, actor_hidden_dims[0]),
                activation,
                *[layer for h_in, h_out in zip(actor_hidden_dims[:-1], actor_hidden_dims[1:])
                  for layer in (nn.Linear(h_in, h_out), activation)],
                nn.Linear(actor_hidden_dims[-1], num_actions)
            )
            for _ in range(num_experts)
        ])

        # create gate network
        gate_layers = []
        gate_layers.append(nn.Linear(num_actor_obs, gating_hidden_dims[0]))
        gate_layers.append(activation)
        for i in range(len(gating_hidden_dims) -1 ):
            gate_layers.append(nn.Linear(gating_hidden_dims[i], gating_hidden_dims[i + 1]))
            gate_layers.append(activation)
        
        gate_layers.append(nn.Linear(gating_hidden_dims[-1], num_experts))

        self.gating_network = nn.Sequential(*gate_layers)

        print(f"Initialized MoEActorCritic with {num_experts} experts.")



    def update_distribution(self, observations):
        gating_logits = self.gating_network(observations)
        gating_weights = F.softmax(gating_logits, dim=-1)
        expert_means = torch.stack([expert(observations) for expert in self.experts], dim=1)
        mean = torch.sum(gating_weights.unsqueeze(-1) * expert_means, dim=1)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)


    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def act_inference(self, observations):
        
        
        gating_logits = self.gating_network(observations)
        gating_weights = F.softmax(gating_logits, dim=-1)
        expert_means = torch.stack([expert(observations) for expert in self.experts], dim=1)

        mean = torch.sum(gating_weights.unsqueeze(-1) * expert_means, dim=1)

        return mean

class MyOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: MoEActorCritic | MyActorCritic | ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(policy,
                                                 device=self.device,
                                                 **self.alg_cfg,
                                                 multi_gpu_cfg=self.multi_gpu_cfg)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs],
                                                                    until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        print(f"--- Policy type: {policy.__class__.__name__} ---")
        if hasattr(policy, 'num_experts'):
            print(f"Number of experts: {policy.num_experts}")

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Call superclass logger (prints losses, mean reward, etc.)
        super().log(locs, width, pad)

        log_string = f"""{'-' * width}\n"""

        # --- append per-task rewards and episode statistics ---
        infos = locs.get("infos", {})
        for task_key, task_data in infos.items():
            if task_key in ["observations", "time_outs"]:
                continue
            if not isinstance(task_data, dict):
                continue

            task_name = task_data.get("task_name", task_key)
            mean_rew = task_data.get("mean_reward", float("nan"))
            mean_ep_rew = task_data.get("mean_episode_reward", float("nan"))
            mean_ep_len = task_data.get("mean_episode_length", float("nan"))

            # Log all metrics in the same aligned style as superclass
            log_string += (
                f"{task_name + ' (per-step):':>{pad}} {mean_rew:8.3f}\n"
                f"{task_name + ' (episode):':>{pad}} {mean_ep_rew:8.3f}\n"
                f"{task_name + ' (ep. length):':>{pad}} {mean_ep_len:8.1f}\n"
            )

            # --- print detailed reward terms if available ---
            log_dict = task_data.get("log", {})
            if log_dict:
                log_string += f"{' ' * (pad - 2)}Reward term breakdown:\n"
                for term_name, term_value in log_dict.items():
                    if torch.is_tensor(term_value):
                        term_value = term_value.item()
                    log_string += f"{' ' * pad}- {term_name:<25}: {term_value:8.4f}\n"

        # --- print final block ---
        print(log_string)

        # --- TensorBoard / WandB logging ---
        if self.writer:
            for task_key, task_data in infos.items():
                if task_key in ["observations", "time_outs"]:
                    continue
                if not isinstance(task_data, dict):
                    continue

                task_name = task_data.get("task_name", task_key)
                mean_rew = task_data.get("mean_reward", float("nan"))
                mean_ep_rew = task_data.get("mean_episode_reward", float("nan"))
                mean_ep_len = task_data.get("mean_episode_length", float("nan"))

                # --- per-step mean reward ---
                self.writer.add_scalar(f"Rewards/{task_name}/mean_per_step", mean_rew, self.current_learning_iteration)

                # --- per-episode mean reward ---
                self.writer.add_scalar(f"Rewards/{task_name}/mean_per_episode", mean_ep_rew, self.current_learning_iteration)

                # --- per-episode mean length ---
                self.writer.add_scalar(f"Rewards/{task_name}/mean_episode_length", mean_ep_len, self.current_learning_iteration)

                # --- log all sub-terms silently (like before) ---
                log_dict = task_data.get("log", {})
                for term_name, term_value in log_dict.items():
                    if torch.is_tensor(term_value):
                        term_value = term_value.item()
                    self.writer.add_scalar(
                        f"Rewards/{task_name}/{term_name}",
                        term_value,
                        self.current_learning_iteration,
                    )

    


    def _log_to_console(self, name: str, value: float, width: int, pad: int):
        print(f"{name:<{pad}} | {value:>{width - pad - 3}.3f}")



    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self._log_moe_diagnostics(obs)


            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))


    def _log_moe_diagnostics(self, obs: torch.Tensor, num_tasks=4):
        """
        Logging function for MoE statistics, if not running MoE, can be ignored or removed
        
        :param self: Description
        :param obs: Description
        :type obs: torch.Tensor
        :param num_tasks: provide for logging the true task number
        """

        if self.current_learning_iteration % 20 != 0:
            return  # skip logging for other iterations


        policy = self.alg.policy

        if not hasattr(policy, "num_experts"):
            return

        if self.writer is None:
            return

        # Create storage directory for raw gating weights
        weights_dir = os.path.join(self.log_dir, "gating_weights")
        os.makedirs(weights_dir, exist_ok=True)

        with torch.no_grad():
            # subsample for speed + stability
            obs = obs[: min(2048, obs.shape[0])]

            # ---- gating ----
            gating_logits = policy.gating_network(obs)
            gating_weights = torch.softmax(gating_logits, dim=-1)  # [B, E]

            # ---- global expert usage ----
            mean_weights = gating_weights.mean(dim=0)
            for i, w in enumerate(mean_weights):
                self.writer.add_scalar(
                    f"MoE/global/expert_{i}",
                    w.item(),
                    self.current_learning_iteration,
                )

            # ---- entropy ----
            entropy = -(gating_weights * torch.log(gating_weights + 1e-8)).sum(dim=-1)
            self.writer.add_scalar(
                "MoE/global/entropy",
                entropy.mean().item(),
                self.current_learning_iteration,
            )

            # ---- histograms ----
            for i in range(policy.num_experts):
                self.writer.add_histogram(
                    f"MoE/global/weights_hist/expert_{i}",
                    gating_weights[:, i],
                    self.current_learning_iteration,
                )

            # =====================================================
            # Task-conditioned gating (from one-hot in observation)
            # =====================================================
            task_one_hot = obs[:, -num_tasks:]
            task_ids = torch.argmax(task_one_hot, dim=-1)

            for t in torch.unique(task_ids):
                mask = task_ids == t
                if mask.sum() < 10:
                    continue  # avoid noise

                task_mean = gating_weights[mask].mean(dim=0)
                for i, w in enumerate(task_mean):
                    self.writer.add_scalar(
                        f"MoE/task_{int(t)}/expert_{i}",
                        w.item(),
                        self.current_learning_iteration,
                    )

                # optional: histogram to see distribution of expert usage for this task
                for i in range(policy.num_experts):
                    self.writer.add_histogram(
                        f"MoE/task_{int(t)}/weights_hist/expert_{i}",
                        gating_weights[mask, i],
                        self.current_learning_iteration,
                    )

                # ----- NEW: store raw gating weights for offline plotting -----
                # filename: task_{t}_iter_{iteration}.npz
                np.savez(
                    os.path.join(
                        weights_dir,
                        f"task_{int(t)}_iter_{self.current_learning_iteration}.npz"
                    ),
                    task_weights=gating_weights[mask].cpu().numpy()
                )

        if self.current_learning_iteration % 10 == 0:
            print(
                "[MoE] mean expert weights:",
                ", ".join(f"{w:.2f}" for w in mean_weights.tolist())
            )