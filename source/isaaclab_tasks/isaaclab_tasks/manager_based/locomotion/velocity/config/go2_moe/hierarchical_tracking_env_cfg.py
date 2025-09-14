import math
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_moe.flat_env_cfg import UnitreeGo2FlatEnvCfg

# ------------------------------------------------------------------
# Load pretrained low-level velocity policy
# ------------------------------------------------------------------
from your_low_level_policy_module import LowLevelPolicy  # replace with actual class

low_level_policy = LowLevelPolicy()
checkpoint = torch.load("path/to/model.pt", map_location="cpu")
low_level_policy.load_state_dict(checkpoint['model_state_dict'])
low_level_policy.eval()  # freeze weights
for param in low_level_policy.parameters():
    param.requires_grad = False

# ------------------------------------------------------------------
# Base locomotion env
# ------------------------------------------------------------------
LOW_LEVEL_ENV_CFG = UnitreeGo2FlatEnvCfg()

# ------------------------------------------------------------------
# Event Config
# ------------------------------------------------------------------
@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )


# ------------------------------------------------------------------
# Observations for high-level policy
# ------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # include robot pose and velocity
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        # optionally include low-level action outputs as high-level observations
        # low_level_action = ObsTerm(func=lambda env: low_level_policy(env.get_high_level_action()))
    policy: PolicyCfg = PolicyCfg()


# ------------------------------------------------------------------
# Rewards
# ------------------------------------------------------------------
@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5, params={"std": 2.0, "command_name": "pose_command"}
    )
    position_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5, params={"std": 0.2, "command_name": "pose_command"}
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2, params={"command_name": "pose_command"}
    )


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------
@configclass
class CommandsCfg:
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)
        ),
    )


# ------------------------------------------------------------------
# Terminations
# ------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


# ------------------------------------------------------------------
# Hierarchical Goal Tracking Env
# ------------------------------------------------------------------
@configclass
class HierarchicalUnitreeGoalTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """
    Hierarchical environment integrating frozen low-level velocity-tracking policy.
    High-level policy outputs goal-tracking commands.
    Low-level policy outputs velocity actions.
    """

    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene.replace(
        robot=UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    )
    actions = LOW_LEVEL_ENV_CFG.actions  # action space of the low-level policy
    observations = ObservationsCfg()
    events = EventCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

    # ------------------------------------------------------------------
    # Override step to integrate low-level policy
    # ------------------------------------------------------------------
    def step(self, high_level_action):
        """
        Step environment using high-level action mapped through low-level policy.
        """
        with torch.no_grad():  # freeze low-level policy
            low_level_action = low_level_policy(high_level_action)

        # Pass low-level action to parent environment
        obs, reward, done, info = super().step(low_level_action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return super().reset(**kwargs)


# ------------------------------------------------------------------
# Optional: smaller "play" version of the environment
# ------------------------------------------------------------------
class HierarchicalUnitreeGoalTrackingEnvCfg_PLAY(HierarchicalUnitreeGoalTrackingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
