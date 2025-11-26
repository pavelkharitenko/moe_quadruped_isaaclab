# legstand_rewards.py
# IsaacLab-style smooth reward shaping for a leg-stand skill.

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# 1. Feet height reward (smooth exponential, no clipping)
# -----------------------------------------------------------------------------
def legstand_feet_height_exp(
        env: ManagerBasedRLEnv,
        std: float,
        target_height: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    height_err = torch.sum((feet_height - target_height)**2, dim=1)
    rew = torch.exp(-height_err / (std * std))  # ∈ (0,1]

    return rew


# -----------------------------------------------------------------------------
# 2. Upright base orientation reward (smooth & bounded)
# -----------------------------------------------------------------------------
def legstand_orientation_exp(
        env: ManagerBasedRLEnv,
        target_gravity: list[float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    pg = asset.data.projected_gravity_b  # gravity direction in base frame

    pg = torch.nan_to_num(pg, 0.0)
    target = torch.tensor(target_gravity, device=env.device)

    err = torch.sum((pg - target)**2, dim=1)
    rew = torch.exp(-2.0 * err)  # smooth [0,1]

    return rew


# -----------------------------------------------------------------------------
# 3. Front feet airborne reward (soft, continuous instead of boolean)
# -----------------------------------------------------------------------------
def legstand_front_feet_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Foot normal forces (positive = contact)
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    # Soft sigmoid: ≈1 when no contact, ≈0 when foot loaded
    no_contact_soft = torch.exp(-0.02 * torch.abs(fz))
    rew = torch.mean(no_contact_soft, dim=1)  # average front feet score

    return rew


# -----------------------------------------------------------------------------
# 4. Rear foot support (soft, not hard contact threshold)
# -----------------------------------------------------------------------------
def legstand_back_feet_support(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    # Rear feet: want positive forces but smoothly
    support = torch.relu(fz)  # keep only positive
    support = torch.sum(support, dim=1)

    # Smooth normalization (keeps it approx. ≤ 1)
    rew = 1.0 - torch.exp(-support * 0.01)

    return rew


# -----------------------------------------------------------------------------
# 5. Base contact (soft penalty, not clipped or hard)
# -----------------------------------------------------------------------------
def legstand_base_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    # Soft penalty: grows smoothly with collision force
    penalty = torch.relu(torch.sum(fz, dim=1))
    rew_penalty = -0.1 * torch.tanh(penalty * 0.01)

    return rew_penalty


# -----------------------------------------------------------------------------
# 6. Drift penalty (convert into reward via smooth exp)
# -----------------------------------------------------------------------------
def legstand_drift_penalty(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    dx = asset.data.root_pos_w[:, 0]
    dy = asset.data.root_pos_w[:, 1]

    dist = torch.sqrt(dx**2 + dy**2)
    rew = torch.exp(-4.0 * dist)

    return rew


# -----------------------------------------------------------------------------
# 7. Bonus upright & front-feet air (soft, not boolean)
# -----------------------------------------------------------------------------
def legstand_bonus_upright(
        env: ManagerBasedRLEnv,
        orientation_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[orientation_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Uprightness in IsaacLab = projected_gravity_b[:,0] close to target +X axis
    upright_raw = asset.data.projected_gravity_b[:, 0]
    upright = torch.sigmoid(5.0 * (upright_raw - 0.7))  # soft reward ∈ (0,1)

    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    front_no_contact = torch.exp(-0.02 * torch.abs(fz))
    front_air_score = torch.mean(front_no_contact, dim=1)

    # Soft multiplicative bonus
    bonus = upright * front_air_score

    return 0.5 * bonus  # scale small so it doesn't dominate
