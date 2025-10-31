import pprint
import torch
from collections import deque
import os
import time
import statistics
from rsl_rl.utils import store_code_state

import rsl_rl
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
        policy: MyActorCritic |ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

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
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
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
        super().log(locs, width, pad)

        log_string = f"""{'-' * width}\n"""

        # --- append per-task mean rewards, same style ---
        infos = locs.get("infos", {})
        for task_key, task_data in infos.items():
            if task_key in ["observations", "time_outs"]:
                continue
            if not isinstance(task_data, dict) or "mean_reward" not in task_data:
                continue
            task_name = task_data.get("task_name", task_key)
            mean_rew = task_data["mean_reward"]
            log_string += f"{task_name + ':':>{pad}} {mean_rew:8.3f}\n"


        # --- print final log block ---
        print(log_string)

        # --- TensorBoard / WandB logging ---
        if self.writer:
            for task_key, task_data in infos.items():
                if task_key in ["observations", "time_outs"]:
                    continue
                if not isinstance(task_data, dict) or "mean_reward" not in task_data:
                    continue
                task_name = task_data.get("task_name", task_key)
                mean_rew = task_data["mean_reward"]
                self.writer.add_scalar(f"Rewards/{task_name}/mean", mean_rew, self.current_learning_iteration)

                # log sub-terms silently
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
