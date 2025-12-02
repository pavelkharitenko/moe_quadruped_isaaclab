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

def legstand_base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward base height near the target.
    Gaussian shape: exp(-(z - target)^2 / sigma^2)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]  # base height

    height_err = (z - target_height) ** 2
    return torch.exp(-height_err / (sigma ** 2))


def legstand_rear_leg_straight_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_angles: dict[str, float] = {
        "RL_hip_joint": 0.1,
        "RR_hip_joint": 0.1,
        #"FL_thigh_joint": -0.8,
        #"FR_thigh_joint": -0.8,
        "RL_thigh_joint": 1.8,
        "RR_thigh_joint": 1.8,
        "RL_calf_joint": -1.2,
        "RR_calf_joint": -1.2,
    },
    sigma: float = 0.3,
) -> torch.Tensor:
    """
    Reward rear legs being close to straight target angles.
    Computes MSE over rear joints and applies exp(-error/sigma^2).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    q = asset.data.joint_pos

    # Build target vector aligned to joint order
    target_q = torch.tensor(
        [target_angles.get(n, 0.0) for n in joint_names],
        device=env.device,
        dtype=torch.float,
    )

    # mask only rear joints
    rear_ids = [i for i, n in enumerate(joint_names)
                if n.startswith("RL_") or n.startswith("RR_")]

    diff = q[:, rear_ids] - target_q[rear_ids]
    mse = torch.mean(diff ** 2, dim=1)

    return torch.exp(-mse / (sigma ** 2))




def legstand_leg_symmetry_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.2,
) -> torch.Tensor:
    """
    Encourage symmetry between left and right joints (front and rear legs).
    Computes mean squared difference and converts via exp(-err/sigma^2).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    q = asset.data.joint_pos

    # --- Rear legs ---
    rl_ids = [i for i, n in enumerate(joint_names) if n.startswith("RL_")]
    rr_ids = [i for i, n in enumerate(joint_names) if n.startswith("RR_")]
    rl = q[:, rl_ids]
    rr = q[:, rr_ids]
    rear_diff = rl - rr

    # --- Front legs ---
    fl_ids = [i for i, n in enumerate(joint_names) if n.startswith("FL_")]
    fr_ids = [i for i, n in enumerate(joint_names) if n.startswith("FR_")]
    fl = q[:, fl_ids]
    fr = q[:, fr_ids]
    front_diff = fl - fr

    # combine errors
    diff = torch.cat([rear_diff, front_diff], dim=1)
    mse = torch.mean(diff ** 2, dim=1)

    return torch.exp(-mse / (sigma ** 2))
