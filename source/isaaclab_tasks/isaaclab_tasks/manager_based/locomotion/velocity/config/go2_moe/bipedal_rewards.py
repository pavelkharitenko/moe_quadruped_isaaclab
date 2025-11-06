# bipedal_rewards.py
# Full implementation of all bipedal reward and regularization terms.

import torch
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


DEFAULT_JOINT_ANGLES_GO2 = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}


def get_default_joint_tensor(asset, device, defaults_dict):
    """Return tensor of default joint angles aligned with the asset's joint order."""
    return torch.tensor(
        [defaults_dict.get(name, 0.0) for name in asset.data.joint_names],
        dtype=torch.float,
        device=device,
    )


# === BIPEDAL STAND ===

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


"""
def bipedal_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    g_proj = asset.data.projected_gravity_b
    theta = torch.arccos(torch.clamp(g_proj[:, 2], -1.0, 1.0))
    return (0.5 * torch.cos(theta) + 0.5) ** 2
"""

def orientation_cosine_reward(env, asset_cfg, target_gravity=[-1,0,0]):
    asset: RigidObject = env.scene[asset_cfg.name]
    g_proj = asset.data.projected_gravity_b
    target = torch.tensor(target_gravity, device=env.device)
    g_proj = g_proj / g_proj.norm(dim=1, keepdim=True)
    target = target / target.norm()
    cos_sim = torch.sum(g_proj * target, dim=1)
    return (0.5 * cos_sim + 0.5) ** 2  # reward in [0,1]

def bipedal_base_height_linear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, Tmin: float, Tmax: float) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    zroot = torch.clamp((z - Tmin) / (Tmax - Tmin), 0, 1)
    return zroot


# === BIPEDAL TRACKING ===
def bipedal_track_lin_vel_exp(env: ManagerBasedRLEnv, target_lin_vel, sigma: float) -> torch.Tensor:
    v = env.scene["robot"].data.root_lin_vel_b
    vel_error = torch.norm(v - torch.tensor(target_lin_vel, device=env.device), dim=1)
    return torch.exp(-vel_error ** 2 / sigma)


def bipedal_track_ang_vel_exp(env: ManagerBasedRLEnv, target_ang_vel, sigma: float) -> torch.Tensor:
    omega = env.scene["robot"].data.root_ang_vel_b
    vel_error = torch.norm(omega - torch.tensor(target_ang_vel, device=env.device), dim=1)
    return torch.exp(-vel_error ** 2 / sigma)


def bipedal_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def bipedal_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.reset_buf.float()


# === BIPEDAL REGULARIZATION ===
def bipedal_rear_legstand(
    env: ManagerBasedRLEnv,
    sensor_cfg_rear: SceneEntityCfg,
    sensor_cfg_front: SceneEntityCfg,
    max_force: float = 100.0,
) -> torch.Tensor:
    """
    Dense rear leg stand reward:
    - Encourages rear legs to carry weight (higher contact forces)
    - Discourages front legs from supporting weight
    """

    scene = env.scene
    rear_sensor: ContactSensor = scene.sensors[sensor_cfg_rear.name]
    front_sensor: ContactSensor = scene.sensors[sensor_cfg_front.name]

    # Extract vertical (z) contact forces
    rear_forces = torch.clamp(rear_sensor.data.net_forces_w[:, sensor_cfg_rear.body_ids, 2], 0.0, max_force)
    front_forces = torch.clamp(front_sensor.data.net_forces_w[:, sensor_cfg_front.body_ids, 2], 0.0, max_force)

    # Normalize summed forces to [0,1]
    rear_support = torch.sum(rear_forces, dim=1) / max_force
    front_support = torch.sum(front_forces, dim=1) / max_force

    # Reward higher when rear carries weight and front is light
    reward = rear_support * (1.0 - torch.clamp(front_support, 0.0, 1.0))

    return torch.clamp(reward, 0.0, 1.0)


def bipedal_rear_leg_straight(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_angles: dict[str, float] = {
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "RL_thigh_joint": 1.2,
        "RR_thigh_joint": 1.2,
        "RL_calf_joint": -2.2,
        "RR_calf_joint": -2.2,
    },
    sigma: float = 0.5,
) -> torch.Tensor:
    """Reward rear legs for being straight (close to target angles).
    Uses a Gaussian shape around desired joint angles for smoothness."""
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    q = asset.data.joint_pos

    # Create tensor of target angles aligned with joint order
    target_q = torch.tensor(
        [target_angles.get(n, 0.0) for n in joint_names],
        device=env.device,
        dtype=torch.float,
    )

    # Select only rear joints
    rear_ids = [i for i, n in enumerate(joint_names) if n.startswith("RL_") or n.startswith("RR_")]

    # Compute squared distance from target
    diff = q[:, rear_ids] - target_q[rear_ids]
    mse = torch.mean(diff ** 2, dim=1)

    # Exponential falloff reward → 1.0 if perfectly aligned, smoothly decays as joints bend
    reward = torch.exp(-mse / sigma)
    return reward




def bipedal_hip_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, side: str) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    q = asset.data.joint_pos
    q_default = get_default_joint_tensor(asset, env.device, DEFAULT_JOINT_ANGLES_GO2)

    hip_ids = [i for i, n in enumerate(joint_names) if n.startswith(side + "L_hip") or n.startswith(side + "R_hip")]

    diff = q[:, hip_ids] - q_default[hip_ids]
    return torch.sum(diff ** 2, dim=1)


def bipedal_rear_pos_balance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize rear leg lateral asymmetry."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_names = asset.data.body_names

    pos = asset.data.body_pos_w
    rear_left_id = body_names.index("RL_foot")
    rear_right_id = body_names.index("RR_foot")

    rear_left = pos[:, rear_left_id, 1]
    rear_right = pos[:, rear_right_id, 1]
    return torch.abs(rear_left - rear_right)

def bipedal_front_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation of front leg joint positions from default."""
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    q = asset.data.joint_pos
    q_default = get_default_joint_tensor(asset, env.device, DEFAULT_JOINT_ANGLES_GO2)

    front_ids = [i for i, n in enumerate(joint_names) if n.startswith("FL_") or n.startswith("FR_")]
    diff = q[:, front_ids] - q_default[front_ids]
    return torch.sum(diff ** 2, dim=1)


def bipedal_front_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities of front legs."""
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    front_ids = [i for i, n in enumerate(joint_names) if n.startswith("FL_") or n.startswith("FR_")]
    qd = asset.data.joint_vel[:, front_ids]
    return torch.sum(qd ** 2, dim=1)


def bipedal_front_joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations of front legs."""
    asset: RigidObject = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    front_ids = [i for i, n in enumerate(joint_names) if n.startswith("FL_") or n.startswith("FR_")]
    qdd = asset.data.joint_acc[:, front_ids]
    return torch.sum(qdd ** 2, dim=1)


def bipedal_legs_energy_substeps(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize mechanical power (torque × velocity)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque
    qd = asset.data.joint_vel
    power = tau * qd
    return torch.sum(torch.abs(power), dim=1)


def bipedal_torque_exceed_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, limit: float) -> torch.Tensor:
    """Penalize torques that exceed motor limits."""
    asset: RigidObject = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque
    over = (torch.abs(tau) > limit).float()
    return torch.sum(over, dim=1)


def bipedal_joint_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint positions near the limits."""
    asset: RigidObject = env.scene[asset_cfg.name]
    q = asset.data.joint_pos                     # [num_envs, num_joints]
    
    # soft_joint_pos_limits: [num_envs, num_joints, 2] -> last dim: [min, max]
    qmin = asset.data.soft_joint_pos_limits[:, :, 0]  # [num_envs, num_joints]
    qmax = asset.data.soft_joint_pos_limits[:, :, 1]  # [num_envs, num_joints]

    near_limit = ((q < qmin + 0.05) | (q > qmax - 0.05)).float()  # [num_envs, num_joints]
    return torch.sum(near_limit, dim=1)    



def bipedal_collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize collisions of non-leg parts."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return torch.clamp(contact, 0, 100).sum(dim=1) / 100.0


""" fix later
def bipedal_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    #Penalize sudden action changes.
    delta_a = env.actions - env.last_actions
    return -torch.sum(delta_a ** 2, dim=1)
"""


def bipedal_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize total joint velocity energy."""
    asset: RigidObject = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel
    return torch.sum(qd ** 2, dim=1)


def bipedal_joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize total joint acceleration energy."""
    asset: RigidObject = env.scene[asset_cfg.name]
    qdd = asset.data.joint_acc
    return torch.sum(qdd ** 2, dim=1)
