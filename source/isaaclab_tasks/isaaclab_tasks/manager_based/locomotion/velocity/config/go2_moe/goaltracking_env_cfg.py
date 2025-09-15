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
    """Reward for aligning robot's heading with the goal direction."""
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute forward vector in XY plane
    # Assuming local X-axis is forward
    quat = asset.data.root_quat_w  # [w, x, y, z]
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Forward vector in world frame
    forward_x = 2 * (qx*qz + qw*qy)
    forward_y = 2 * (qy*qz - qw*qx)
    forward_vec = torch.stack([forward_x, forward_y], dim=1)
    forward_vec = torch.nn.functional.normalize(forward_vec, dim=1)

    pos_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.extras["goal_xy"].to(env.device)
    goal_dir = torch.nn.functional.normalize(goal_xy - pos_xy, dim=1)

    heading_alignment = torch.sum(forward_vec * goal_dir, dim=1)  # cos(theta)
    return heading_alignment  # +1 = perfect alignment, -1 = opposite


def forward_toward_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving toward the goal (velocity alignment)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]
    vel_xy = asset.data.root_vel_w[:, :2]
    goal_xy = env.extras["goal_xy"].to(env.device)
    direction = torch.nn.functional.normalize(goal_xy - pos_xy, dim=1)
    return torch.sum(vel_xy * direction, dim=1)  # reward for velocity toward goal


def upright_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for excessive roll or pitch to discourage rolling."""
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Compute roll and pitch
    roll = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    pitch = torch.asin(torch.clamp(2*(qw*qy - qz*qx), -1.0, 1.0))
    return - (torch.abs(roll) + torch.abs(pitch))  # negative = penalty


# ----------------------------
# Goal sampling event
# ----------------------------

def sample_new_goal(env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg) -> None:
    """Samples a new XY goal and sets goal_yaw toward the goal direction."""
    num_envs = env.scene.num_envs
    device = env.device

    # Sample random XY goal within [-3, 3] range
    goal_xy = (torch.rand((num_envs, 2), device=device) * 6.0) - 3.0

    # Use asset_cfg.name to access robot
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]

    # Compute yaw toward goal
    goal_dir = goal_xy - pos_xy
    goal_yaw = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])

    # Store in env.extras
    env.extras["goal_xy"] = goal_xy
    env.extras["goal_yaw"] = goal_yaw


# ----------------------------
# Configs
# ----------------------------



@configclass
class GoalTrackingEventsCfg(EventCfg):

    sample_goal = EventTerm(
        func=sample_new_goal,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),  # optional parameter with default
        },
    )

@configclass
class GoalTrackingRewardsCfg(RewardsCfg):
    """Custom rewards for goal tracking with heading and stability."""

    goal_distance = RewTerm(
        func=goal_distance_exp,
        weight=2.0,
        params={"std": 1.0},
    )

    goal_heading = RewTerm(
        func=goal_heading_error,
        weight=1.5,  # stronger than before
    )

    move_forward = RewTerm(
        func=forward_toward_goal,
        weight=1.0,
    )

    upright = RewTerm(
        func=upright_penalty,
        weight=1.5,
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
