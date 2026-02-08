import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import os
import datetime
import time

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

# DPMM-VAE imports
from isaaclab_rl.rsl_rl.DPMM.dpmm_utils import (DPMMReplayBuffer, Transition, LatentInjectedEnvWrapper, class_to_dict)

from isaaclab_rl.rsl_rl.DPMM.dpmm_config import DpmmVaeCfgBnpy

from isaaclab_rl.rsl_rl.DPMM.MELTS.tigr.task_inference.prediction_networks import DecoderMDP
from isaaclab_rl.rsl_rl.DPMM.MELTS.tigr.task_inference.dpmm_bnp import BNPModel
from isaaclab_rl.rsl_rl.DPMM.MELTS.tigr.task_inference.dpmm_inference import DecoupledEncoder
from isaaclab_rl.rsl_rl.DPMM.MELTS.tigr.trainer.dpmm_trainer import AugmentedTrainer


class DPMMRunnerConditioned(OnPolicyRunner):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # load DPMM Cfg
        dpmm_cfg = DpmmVaeCfgBnpy
        self.dpmm_cfg = dpmm_cfg

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

        # add latent dimension encoding to obs_dim:

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
            num_obs + self.dpmm_cfg.z_dim,  # account for latent z dim
            num_privileged_obs + self.dpmm_cfg.z_dim,  # account for latent z dim
            self.env.num_actions,
            **self.policy_cfg).to(self.device)

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
            [num_obs + self.dpmm_cfg.z_dim],  # PPO will be conditioned on latent z, so add z dim (for PPO only)
            [num_privileged_obs + self.dpmm_cfg.z_dim],  # same for priv. obs
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

        # DPMM-VAE initialization
        # --- Context encoding ---
        """
        self.z_dim = encoder_cfg["z_dim"]
        self.current_z = torch.zeros(env.num_envs, self.z_dim, device=device)
        """

        self.dpmm_total_update_steps = 0

        absolute_log_dir = os.path.abspath(  # logdir for DPMM-VAE training
            f"./logs/dpmm/dpmm_logs_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # DPMM-VAE buffer
        self.dpmm_buffer = DPMMReplayBuffer(size=self.dpmm_cfg.dpmm_buffer_size,
                                            nw=self.dpmm_cfg.context_length,
                                            device=self.device)

        # Initialize Bayesian Nonparametric Model (bnpy)
        bnp_model = BNPModel(
            save_dir=absolute_log_dir,
            gamma0=dpmm_cfg.bnp_model.gamma0,
            num_lap=dpmm_cfg.bnp_model.num_lap,
            start_epoch=dpmm_cfg.bnp_model.start_epoch,
            fit_interval=dpmm_cfg.bnp_model.fit_interval,
            birth_kwargs=dict(
                b_startLap=dpmm_cfg.bnp_model.birth.start_lap,
                b_stopLap=dpmm_cfg.bnp_model.birth.stop_lap,
                b_Kfresh=dpmm_cfg.bnp_model.birth.k_fresh,
                b_minNumAtomsForNewComp=dpmm_cfg.bnp_model.birth.min_num_atoms_for_new_comp,
                b_minNumAtomsForTargetComp=dpmm_cfg.bnp_model.birth.min_num_atoms_for_target_comp,
                b_minNumAtomsForRetainComp=dpmm_cfg.bnp_model.birth.min_num_atoms_for_retain_comp,
                b_minPercChangeInNumAtomsToReactivate=dpmm_cfg.bnp_model.birth.min_perc_change_to_reactivate,
                b_debugOutputDir=dpmm_cfg.bnp_model.birth.debug_output_dir,
                b_debugWriteHTML=dpmm_cfg.bnp_model.birth.debug_write_html,
            ),
            merge_kwargs=dict(
                m_startLap=dpmm_cfg.bnp_model.merge.start_lap,
                m_maxNumPairsContainingComp=dpmm_cfg.bnp_model.merge.max_num_pairs_containing_comp,
                m_nLapToReactivate=dpmm_cfg.bnp_model.merge.n_lap_to_reactivate,
                m_pair_ranking_procedure=dpmm_cfg.bnp_model.merge.pair_ranking_procedure,
                m_pair_ranking_direction=dpmm_cfg.bnp_model.merge.pair_ranking_direction,
            ),
        )

        # Derive dimensions from runner / env
        obs_dim = num_obs  # from env.get_observations(), obs for encoder are without latent dim
        action_dim = self.env.num_actions
        reward_dim = 1  # scalar reward (RL assumption)
        state_dim = obs_dim  # state == observation
        tasks_num = self.dpmm_cfg.num_tasks

        transition_dim = (
            state_dim +  # s_t
            action_dim +  # a_t
            reward_dim +  # r_t
            state_dim  # s_{t+1}
        )

        shared_dim = self.dpmm_cfg.shared_dim

        # Encoder
        encoder = DecoupledEncoder(
            shared_dim=shared_dim,
            encoder_input_dim=dpmm_cfg.time_steps * transition_dim,
            latent_dim=dpmm_cfg.z_dim,
            num_classes=tasks_num,
            time_steps=dpmm_cfg.time_steps,
            encoding_mode="trajectory",
            timestep_combination="multiplication",
            encoder_type="gru",
            bnp_model=bnp_model,
        )

        self.encoder = encoder

        # Decoder
        decoder = DecoderMDP(
            action_dim=action_dim,
            state_dim=state_dim,
            reward_dim=reward_dim,
            z_dim=dpmm_cfg.z_dim,
            net_complex=2,
            state_reconstruction_clip=state_dim,
        )

        encoder.to(self.device)
        decoder.to(self.device)

        self.dpmm_trainer = AugmentedTrainer(
            encoder=encoder,
            decoder=decoder,
            replay_buffer=self.dpmm_buffer,  # NOTE: streaming buffer
            replay_buffer_augmented=None,
            batch_size=dpmm_cfg.batch_size,
            num_classes=tasks_num,
            latent_dim=dpmm_cfg.z_dim,
            timesteps=dpmm_cfg.time_steps,
            lr_encoder=dpmm_cfg.trainer.lr_encoder,
            lr_decoder=dpmm_cfg.trainer.lr_decoder,
            alpha_kl_z=dpmm_cfg.trainer.alpha_kl_z,
            beta_euclid=dpmm_cfg.trainer.beta_euclid,
            gamma_sparsity=1e-3,
            regularization_lambda=0.1,
            use_state_diff=False,
            state_reconstruction_clip=state_dim,
            use_data_normalization=True,
            train_val_percent=1.0,
            eval_interval=50,
            early_stopping_threshold=500,
            experiment_log_dir=absolute_log_dir,
            use_regularization_loss=True,
            use_PCGrad=False,
            PCGrad_option="true_task",
            optimizer_class=torch.optim.Adam,
            device=self.device,
            log_dir=self.log_dir,
        )

        # for conditioning PPO: track last context_length-steps, pass to encoder before PPO rollout stage
        self.context_buffer = torch.zeros(
            self.env.num_envs,
            self.dpmm_cfg.time_steps,
            transition_dim,
            device=self.device,
        )
        self.context_ptr = torch.zeros(
            self.env.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        # select subset of envs that contribute to dpmm buffer
        self.dpmm_selected_sample_envs = self.select_random_envs()

    def select_random_envs(self):
        """
        Reselect subenvs from total available envs to fill DPMM's Off-Policy Buffer
        """
        frac = self.dpmm_cfg.num_envs_per_iter
        total_envs = self.env.num_envs

        # Compute number of envs to sample
        n = max(1, int(round(frac * total_envs)))
        n = min(n, total_envs)

        # Randomly select env indices
        env_indices = torch.randperm(total_envs, device=self.device)[:n]

        print(f"selected {n}/{total_envs} envs:", env_indices)
        return env_indices

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
            log_string += (f"{task_name + ' (per-step):':>{pad}} {mean_rew:8.3f}\n"
                           f"{task_name + ' (episode):':>{pad}} {mean_ep_rew:8.3f}\n"
                           f"{task_name + ' (ep. length):':>{pad}} {mean_ep_len:8.1f}\n")

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
                self.writer.add_scalar(f"Rewards/{task_name}/mean_per_episode", mean_ep_rew,
                                       self.current_learning_iteration)

                # --- per-episode mean length ---
                self.writer.add_scalar(f"Rewards/{task_name}/mean_episode_length", mean_ep_len,
                                       self.current_learning_iteration)

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

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
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
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

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

        # one temporary trajectory per env
        self._dpmm_trajs = [[] for _ in range(self.env.num_envs)]
        prev_obs = obs.clone()  # save as "s" in first tuple (s,a,r,s')

        # === Training loop ===
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start_time = time.time()

            # === Collect rollouts ===
            with torch.inference_mode():

                # first todo for conditioned PPO: inject latent z to PPO before rollout

                z, assignments = self.encoder(
                    self.context_buffer)  # we dont need to clone() the buffer, since we use torch.inference_mode()
                #self.alg.set_latent(z)  # now pi(. |z)

                for _ in range(self.num_steps_per_env):

                    # append latent z every step to obs, so pi(a|s) => pi(a|s,z)
                    obs_z = torch.cat([obs, z], dim=-1)  # now combine with the latent encoding z
                    privileged_obs_z = torch.cat([privileged_obs, z], dim=-1)

                    # sample actions
                    actions = self.alg.act(obs_z, privileged_obs_z)

                    # step environment & move to device
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # for PPO inferring the DPMM-VAE later: remeber last (context_length)-transitions:
                    transition_context = torch.cat(
                        [
                            prev_obs,  # [N, Ds]
                            actions,  # [N, Da]
                            rewards.unsqueeze(-1),  # [N, 1]
                            obs,  # [N, Ds]
                        ],
                        dim=-1)

                    idx = self.context_ptr  # [N]
                    self.context_buffer[torch.arange(self.env.num_envs), idx] = transition_context
                    self.context_ptr = (self.context_ptr + 1) % self.dpmm_cfg.time_steps  # increase

                    # for DPMM Buffer: append for i-th env the i-th (s,a,r,s') tuple each step
                    # first check if new subenvs should be sampled from tasks
                    # TODO flush remaining trajectories or set _dpm_trajs list empty when resetting
                    if self.dpmm_total_update_steps % self.dpmm_cfg.reselect_envs_interval == 0:
                        self.dpmm_selected_sample_envs = self.select_random_envs()

                    next_obs = obs.clone()

                    for i in self.dpmm_selected_sample_envs:
                        task_onehot = prev_obs[i][-self.dpmm_cfg.num_tasks:]
                        self._dpmm_trajs[i].append(
                            Transition(
                                obs=prev_obs[i],
                                action=actions[i],
                                reward=rewards[i].view(1),
                                next_obs=next_obs[i],
                                done=dones[i].view(1),
                                task_id=task_onehot,  # or infos[i].get("task_id")
                            ))

                    prev_obs = next_obs.clone()

                    # flush completed/long trajectories into replay buffer
                    flush_list = []

                    for i in self.dpmm_selected_sample_envs:
                        if dones[i] or len(self._dpmm_trajs[i]) >= self.dpmm_cfg.max_traj_len:
                            flush_list.append(self._dpmm_trajs[i])
                            self._dpmm_trajs[i] = []

                    # shuffle trajectories so short ones are not always appended first into DPMM-Buffer
                    if len(flush_list) > 0:
                        perm = torch.randperm(len(flush_list))
                        for p in perm:
                            traj = flush_list[p]
                            for transition in traj:
                                self.dpmm_buffer.buffer.append(transition)

                    obs = self.obs_normalizer(obs)
                    privileged_obs = (self.privileged_obs_normalizer(infos["observations"][self.privileged_obs_type].to(
                        self.device)) if self.privileged_obs_type is not None else obs)

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

                    # increment timesteps collected in env
                    self.dpmm_total_update_steps += 1

                collection_time = time.time() - start_time

                # Compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs_z)

            # log DPMM-buffer statistics
            if len(self.dpmm_buffer) > 0:
                print(f"[DPMM Buffer] size={len(self.dpmm_buffer.buffer)} | "
                      f"last_reward={self.dpmm_buffer.buffer[-1].reward} | "
                      f"done={self.dpmm_buffer.buffer[-1].done}")
                print("dpmm total update steps", self.dpmm_total_update_steps)
                #done_count = sum(t.done for t in self.dpmm_buffer.buffer)
                #print(f"DPMM done ratio = {done_count / len(self.dpmm_buffer.buffer)}")
                #print(f"DPMM done count = {done_count}")

            # DPMM-VAE training update:

            # sample batch from DPMM-Buffer according strategy Sc and pass sequentially to GRU:
            #batch_dpmm = self.dpmm_buffer.sample_contexts(batch_size=self.dpmm_cfg.batch_size,nw=self.dpmm_cfg.context_length)

            # update VAE's KL beta
            if False and len(self.dpmm_buffer
                             ) > self.dpmm_cfg.context_length and it > 40 and it % 4 == 0:  # training disabled for now

                vae_beta = min(self.dpmm_cfg.warmup.beta_final,
                               self.dpmm_cfg.warmup.beta_final * it / self.dpmm_cfg.warmup.warmup_epochs)
                self.dpmm_trainer.alpha_kl_z = vae_beta

                self.dpmm_trainer.train(mixture_steps=self.dpmm_cfg.trainer.mixture_steps, current_epoch=it)

            # === PPO update ===
            start_update = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - start_update
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # log to tensorboard or other
                self.log(locals())

            # === Save model occasionally ===
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        # === Final save ===
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
