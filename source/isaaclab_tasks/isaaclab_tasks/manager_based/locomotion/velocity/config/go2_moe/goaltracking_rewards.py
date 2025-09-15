# goal_tracking_rewards.py
# Reward functions for teaching a quadruped to track a goal
# using IsaacLab's Manager-based API.

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv


# goaltracking_rewards.py
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def goal_distance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for minimizing distance to a goal position in XY."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # Current robot position (XY only)
    pos_xy = asset.data.root_pos_w[:, :2]
    # Goal position stored in env.extras["goal_xy"]
    goal_xy = env.extras["goal_xy"].to(env.device)
    dist = torch.linalg.norm(pos_xy - goal_xy, dim=1)
    return torch.exp(-dist**2 / (2 * std**2))


def goal_heading_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for deviating from desired heading at the goal."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # Current base yaw from quaternion
    yaw = asset.data.root_quat_w[:, [0, 3]]  # simplified yaw extraction
    # Desired yaw (stored in env.extras["goal_yaw"])
    goal_yaw = env.extras["goal_yaw"].to(env.device)
    return -torch.abs(yaw[:, 0] - goal_yaw)  # simple abs error, could wrap to [-pi, pi]


def goal_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shaped distance reward: coarse (far) + fine (near)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_pos = env.command_manager.get_command("goal_position")[:, :2]
    current_pos = asset.data.root_pos_w[:, :2]
    delta = goal_pos - current_pos
    distance = torch.norm(delta, dim=1)

    reward_coarse = 1.0 - torch.tanh(distance / 2.0)
    reward_fine = 1.0 - torch.tanh(distance / 0.2)
    return reward_coarse + reward_fine


def goal_heading(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for aligning base heading toward the goal."""
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_pos = env.command_manager.get_command("goal_position")[:, :2]
    current_pos = asset.data.root_pos_w[:, :2]
    delta = goal_pos - current_pos

    forward = asset.data.forward_w[:, :2]  # world-frame forward vector
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    desired = torch.atan2(delta[:, 1], delta[:, 0])
    heading_error = torch.remainder(desired - heading + torch.pi, 2 * torch.pi) - torch.pi
    return torch.exp(-heading_error**2 / 0.5)


def goal_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Bonus reward if robot reaches goal, stops, stays upright, and at correct height."""
    asset: RigidObject = env.scene[asset_cfg.name]

    goal_pos = env.command_manager.get_command("goal_position")[:, :2]
    current_pos = asset.data.root_pos_w[:, :2]
    delta = goal_pos - current_pos
    distance = torch.norm(delta, dim=1)

    at_goal = (distance < 0.3).float()

    stop = torch.exp(-torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1) / 0.2)

    upright_score = torch.exp(
        -torch.sum(asset.data.projected_gravity_b[:, :2] ** 2, dim=1) / 0.05
    )

    base_height = torch.mean(asset.data.root_pos_w[:, 2].unsqueeze(1) - asset.data.body_pos_w[:, :, 2], dim=1)
    height_error = (base_height - env.cfg.rewards.base_height_target) ** 2
    height_score = torch.exp(-height_error / 0.01)

    return at_goal * stop * upright_score * height_score


def goal_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving velocity in the direction of goal."""
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_pos = env.command_manager.get_command("goal_position")[:, :2]
    current_pos = asset.data.root_pos_w[:, :2]
    delta = goal_pos - current_pos
    distance = torch.norm(delta, dim=1)

    velocity = asset.data.root_lin_vel_w[:, :2]
    velocity_to_goal = torch.sum(velocity * delta / (distance.unsqueeze(1) + 1e-6), dim=1)
    return velocity_to_goal


def base_orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from upright orientation (projected gravity)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return -torch.sum(asset.data.projected_gravity_b[:, :2] ** 2, dim=1)


def base_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from target base height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = torch.mean(asset.data.root_pos_w[:, 2].unsqueeze(1) - asset.data.body_pos_w[:, :, 2], dim=1)
    error = (base_height - env.cfg.rewards.base_height_target) ** 2
    return -error

