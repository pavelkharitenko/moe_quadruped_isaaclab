# legstand_rewards.py
# Reward functions for teaching a quadruped to perform a legstand
# using IsaacLab's Manager-based API.

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv



def legstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height_error = torch.sum(torch.square(feet_height - target_height), dim=1)
    return torch.exp(-feet_height_error / std**2)


def legstand_orientation_l2(
    env: ManagerBasedRLEnv,
    target_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation of robot's base orientation from target (projected gravity).
    For legstand we want base upright, so use target_gravity=[1, 0, 0] or similar."""
    
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.tensor(target_gravity, device=env.device)
    
    return torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)


def legstand_front_feet_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward when both front feet (FL, FR) are in the air (no contact)."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    return torch.all(first_air, dim=1).float()


def legstand_back_feet_support(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Encourage rear legs (RL, RR) to carry weight via contact forces."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return torch.sum(torch.clamp(contact_forces, 0, 100), dim=1) / 100.0


def legstand_base_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize if the robot's base collides with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return -torch.clamp(contact, min=0.0, max=100.0).sum(dim=1) / 100.0

"""
def legstand_joint_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    #Penalize joint positions being close to or beyond their limits.
    asset: RigidObject = env.scene[asset_cfg.name]
    dof_mid = (asset.data.soft_joint_pos_limits[:, 0] + asset.data.soft_joint_pos_limits[:, 1]) * 0.5
    dof_range = (asset.data.soft_joint_pos_limits[:, 1] - asset.data.soft_joint_pos_limits[:, 0]) * 0.5
    distance = torch.abs(asset.data.joint_pos - dof_mid) / (dof_range + 1e-6)
    return torch.sum(distance ** 5, dim=1) / asset.num_dof
"""


def legstand_drift_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize drifting away from the starting XY position (helps balance stability)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    dx = asset.data.root_pos_w[:, 0]
    dy = asset.data.root_pos_w[:, 1]
    return torch.sqrt(dx**2 + dy**2)


def legstand_bonus_upright(
    env: ManagerBasedRLEnv,
    orientation_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Bonus reward if robot is upright AND front feet are in the air."""
    asset: RigidObject = env.scene[orientation_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    upright = asset.data.projected_gravity_b[:, 0] > -0.95
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    front_air = torch.all(first_air, dim=1)

    return (upright & front_air).float() * 2.0
