# bipedal_env_cfg.py
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
import math
from .flat_env_cfg import UnitreeGo2FlatEnvCfg
from .bipedal_rewards import *
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg


@configclass
class UnitreeGo2BipedalRewardsCfg(RewardsCfg):
    # === STAND ===
    orientation = RewTerm(func=bipedal_orientation, weight=3.0, params={"asset_cfg": SceneEntityCfg("robot")})
    base_height_linear = RewTerm(
        func=bipedal_base_height_linear,
        weight=1.8,
        params={"asset_cfg": SceneEntityCfg("robot"), "Tmin": 0.15, "Tmax": 0.35},
    )

    # === TRACKING ===
    track_lin_vel = RewTerm(
        func=bipedal_track_lin_vel_exp,
        weight=3.0,
        params={"target_lin_vel": [0.5, 0.0, 0.0], "sigma": 0.25},
    )
    track_ang_vel = RewTerm(
        func=bipedal_track_ang_vel_exp,
        weight=2.5,
        params={"target_ang_vel": [0.0, 0.0, 0.2], "sigma": 0.25},
    )
    termination = RewTerm(func=bipedal_termination, weight=-1.0)
    alive = RewTerm(func=bipedal_alive, weight=1.0)

    # === REGULARIZATION ===
    rear_air = RewTerm(
        func=bipedal_rear_air,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"])},
    )
    front_hip_pos = RewTerm(
        func=bipedal_hip_pos,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "side": "F"},
    )
    rear_hip_pos = RewTerm(
        func=bipedal_hip_pos,
        weight=-0.18,
        params={"asset_cfg": SceneEntityCfg("robot"), "side": "R"},
    )
    rear_pos_balance = RewTerm(
        func=bipedal_rear_pos_balance,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    front_joint_pos = RewTerm(
        func=bipedal_front_joint_pos,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    front_joint_vel = RewTerm(
        func=bipedal_front_joint_vel,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    front_joint_acc = RewTerm(
        func=bipedal_front_joint_acc,
        weight=-2e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    legs_energy_substeps = RewTerm(
        func=bipedal_legs_energy_substeps,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_exceed_limits = RewTerm(
        func=bipedal_torque_exceed_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit": 30.0},
    )
    joint_limits = RewTerm(func=bipedal_joint_limits, weight=-0.06, params={"asset_cfg": SceneEntityCfg("robot")})
    collision = RewTerm(
        func=bipedal_collision,
        weight=-0.06,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"])},
    )
    joint_vel = RewTerm(func=bipedal_joint_vel, weight=-2e-3, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_acc = RewTerm(func=bipedal_joint_acc, weight=-3e-6, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class UnitreeGo2BipedalEnvCfg(UnitreeGo2FlatEnvCfg):
    rewards: UnitreeGo2BipedalRewardsCfg = UnitreeGo2BipedalRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Flat terrain for balance learning
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2BipedalEnvCfg_PLAY(UnitreeGo2BipedalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
