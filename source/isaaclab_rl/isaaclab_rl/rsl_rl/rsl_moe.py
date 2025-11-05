import torch
from collections import deque
import os
import time
import statistics
import pandas as pd
import csv

import rsl_rl
from rsl_rl.utils import store_code_state
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
        # add your custom layers or logic here
        print("#################################################")
        print(" Using Custom MyActorCritic")
        print("#################################################")


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
        policy: MyActorCritic | ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
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

    def learn1(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Extended learn() with full diagnostics for PPO buffer and reward statistics."""

        # === Initialize logger ===
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
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
                raise ValueError("Logger type not found. Choose 'neptune', 'wandb' or 'tensorboard'.")

        # === Sanity print ===
        print(f"\n--- PPO Learn Initialization ---")
        print(f"Device: {self.device}")
        print(f"num_envs: {self.env.num_envs}, num_steps_per_env: {self.num_steps_per_env}")
        print(f"Total rollout batch size: {self.env.num_envs * self.num_steps_per_env}")
        if hasattr(self.alg, "storage"):
            try:
                obs_shape = getattr(self.alg.storage, "observations", torch.empty(0)).shape
                print(f"Storage observations shape: {obs_shape}")
            except Exception as e:
                print(f"(Could not access storage shape: {e})")
        print(f"num_learning_epochs: {getattr(self.alg, 'num_learning_epochs', 'N/A')}")
        print(f"num_mini_batches: {getattr(self.alg, 'num_mini_batches', 'N/A')}")
        print(f"---------------------------------\n")

        # === Randomize episode starts (optional) ===
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # === Get initial observations ===
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        # === Initialize buffers ===
        ep_infos = []
        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # === CSV setup ===
        csv_path = os.path.join(self.log_dir, "training_diagnostics.csv")
        write_header = not os.path.exists(csv_path)

        # === Training loop ===
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start_time = time.time()

            # === Collect rollouts ===
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)


                    
                    # Reward statistics
                    #if it < 5 or it % 10 == 0:
                    #    print(f"[it {it}] reward mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
                    #        f"min={rewards.min():.3f}, max={rewards.max():.3f}")

                    obs = self.obs_normalizer(obs)
                    privileged_obs = (self.privileged_obs_normalizer(
                        infos["observations"][self.privileged_obs_type].to(self.device)
                    ) if self.privileged_obs_type is not None else obs)

                    # Store step
                    self.alg.process_env_step(rewards, dones, infos)

                    # Reward aggregation
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    done_ids = (dones > 0).nonzero(as_tuple=False)
                    if len(done_ids) > 0:
                        rewbuffer.extend(cur_reward_sum[done_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[done_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[done_ids] = 0
                        cur_episode_length[done_ids] = 0

                collection_time = time.time() - start_time

                # Compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # === PPO update ===
            start_update = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - start_update
            self.current_learning_iteration = it

            # === Diagnostics: PPO internals ===
            if hasattr(self.alg, "storage"):
                adv = getattr(self.alg.storage, "advantages", None)
                if adv is not None:
                    print(f"[it {it}] Advantage mean={adv.mean():.4f}, std={adv.std():.4f}, "
                        f"min={adv.min():.3f}, max={adv.max():.3f}")
                values = getattr(self.alg.storage, "values", None)
                if values is not None:
                    print(f"[it {it}] Value mean={values.mean():.4f}, std={values.std():.4f}")

            # === Prepare metrics ===
            def safe(x):
                return x.item() if torch.is_tensor(x) else float(x)

            metrics = {
                "iteration": it,
                "num_envs": self.env.num_envs,
                "steps_per_env": self.num_steps_per_env,
                "collection_time": collection_time,
                "learn_time": learn_time,
                "entropy": safe(loss_dict.get("entropy", float("nan"))),
                "policy_loss": safe(loss_dict.get("surrogate", float("nan"))),
                "value_loss": safe(loss_dict.get("value_function", float("nan"))),
                "learning_rate": safe(getattr(self.alg.optimizer.param_groups[0], "lr", float("nan"))),
                "mean_reward": statistics.mean(rewbuffer) if len(rewbuffer) > 0 else float("nan"),
                "mean_ep_len": statistics.mean(lenbuffer) if len(lenbuffer) > 0 else float("nan"),
            }

            # === Write metrics to CSV ===
            if self.log_dir is not None and not self.disable_logs:
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(metrics)

                # log to tensorboard or other
                self.log(locals())

            # === Print periodic buffer sanity check ===
            if it % 20 == 0:
                total_expected = self.env.num_envs * self.num_steps_per_env
                if hasattr(self.alg.storage, "observations"):
                    obs_buf = self.alg.storage.observations
                    print(f"[it {it}] storage.obs shape={tuple(obs_buf.shape)}, expected batch={total_expected}")
                print(f"[it {it}] mean reward buffer={metrics['mean_reward']:.3f}, mean value_loss={metrics['value_loss']:.3f}")

            # === Save model occasionally ===
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        # === Final save ===
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))


    def _log_to_console(self, name: str, value: float, width: int, pad: int):
        print(f"{name:<{pad}} | {value:>{width - pad - 3}.3f}")


