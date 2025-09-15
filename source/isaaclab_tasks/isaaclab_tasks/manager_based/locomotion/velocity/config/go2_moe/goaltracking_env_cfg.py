# goaltracking_env_cfg.py
# Navigation-like environment for Unitree Go2 to track a random XY + yaw goal
# Built in the style of legstand_env_cfg.py but for goal tracking.

import math
import torch
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg, EventCfg

from isaaclab.managers import SceneEntityCfg, EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject

from .flat_env_cfg import UnitreeGo2FlatEnvCfg


# ----------------------------
# Custom reward functions
# ----------------------------

def goal_distance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for minimizing distance to a goal position in XY."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.extras["goal_xy"].to(env.device)
    dist = torch.linalg.norm(pos_xy - goal_xy, dim=1)
    return torch.exp(-dist**2 / (2 * std**2))


def goal_heading_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for deviating from desired yaw at the goal."""
    asset: RigidObject = env.scene[asset_cfg.name]

    # Extract yaw from quaternion (simplified)
    qw, qz = asset.data.root_quat_w[:, 0], asset.data.root_quat_w[:, 3]
    yaw = 2 * torch.atan2(qz, qw)

    goal_yaw = env.extras["goal_yaw"].to(env.device)
    yaw_error = torch.remainder(yaw - goal_yaw + math.pi, 2 * math.pi) - math.pi
    return -torch.abs(yaw_error)


# ----------------------------
# Goal sampling event
# ----------------------------

def sample_new_goal(env: ManagerBasedRLEnv, assetcfg) -> None:
    """Samples a new XY + yaw goal and stores it in env.extras."""
    num_envs = env.scene.num_envs
    device = env.device

    goal_xy = (torch.rand((num_envs, 2), device=device) * 6.0) - 3.0  # [-3, 3] range
    goal_yaw = (torch.rand((num_envs,), device=device) * 2 * math.pi) - math.pi

    env.extras["goal_xy"] = goal_xy
    env.extras["goal_yaw"] = goal_yaw

    print("##############################################################")
    print("New sampled goal", goal_xy)


# ----------------------------
# Configs
# ----------------------------

@configclass
class GoalTrackingRewardsCfg(RewardsCfg):
    """Custom rewards for goal tracking."""

    # Add distance-to-goal reward
    goal_distance = RewTerm(
        func=goal_distance_exp,
        weight=2.0,
        params={"std": 1.0},
    )

    # Add heading alignment reward
    goal_heading = RewTerm(
        func=goal_heading_error,
        weight=0.5,
    )


@configclass
class GoalTrackingEventsCfg(EventCfg):

    sample_goal = EventTerm(
        func=sample_new_goal,
        mode="reset",
    )


@configclass
class UnitreeGo2GoalTrackingEnvCfg(UnitreeGo2FlatEnvCfg):
    """Unitree Go2 environment for reaching random XY + yaw goals."""

    rewards: GoalTrackingRewardsCfg = GoalTrackingRewardsCfg()
    events: GoalTrackingEventsCfg = GoalTrackingEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Disable velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0

        # Keep useful base/joint/feet rewards from FlatEnvCfg
        # (feet air time, joint limits, collision penalties, etc.)

        # Use flat terrain for goal reaching
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Disable height scanning
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None


# ----------------------------
# PLAY variant
# ----------------------------

class UnitreeGo2GoalTrackingEnvCfg_PLAY(UnitreeGo2GoalTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable corruption
        self.observations.policy.enable_corruption = False

        # Remove random pushes
        self.events.base_external_force_torque = None
        self.events.push_robot = None
