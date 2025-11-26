# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from .legstand_reworked_rewards import *
from isaaclab.managers import RewardTermCfg as RewTerm
import math

from .flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg


@configclass
class UnitreeGo2LegStandRewardsCfg(RewardsCfg):

    legstand_feet_height_exp = RewTerm(
        func=legstand_feet_height_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot"]),
            "target_height": 0.5,
            "std": math.sqrt(0.25)
        },
    )

    legstand_orientation_exp = RewTerm(
        func=legstand_orientation_exp,
        weight=0.0,
        params={"target_gravity": [-1.0, 0.0, 0.0]},  # adjust depending on base-up direction
    )

    legstand_front_feet_air = RewTerm(
        func=legstand_front_feet_air,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot"])},
    )

    legstand_back_feet_support = RewTerm(
        func=legstand_back_feet_support,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"])},
    )

    legstand_base_contact_penalty = RewTerm(
        func=legstand_base_contact_penalty,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"])},
    )

    legstand_drift_penalty = RewTerm(
        func=legstand_drift_penalty,
        weight=0.0,
        params={},
    )

    legstand_bonus_upright = RewTerm(
        func=legstand_bonus_upright,
        weight=0.0,
        params={
            "orientation_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot"]),
        },
    )


@configclass
class UnitreeGo2LegStandReworkedEnvCfg(UnitreeGo2FlatEnvCfg):

    # add custom rewards
    rewards: UnitreeGo2LegStandRewardsCfg = UnitreeGo2LegStandRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable velocity tracking rewards
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0

        # ----------- legstand rewards -----------
        self.rewards.legstand_feet_height_exp.weight = 2.0  # exp reward, main shaping
        self.rewards.legstand_orientation_exp.weight = -3.0  # small pose stabilization
        self.rewards.legstand_front_feet_air.weight = 4.0  # binary air-of-front-feet
        self.rewards.legstand_back_feet_support.weight = 1.0  # small shaping term
        self.rewards.legstand_base_contact_penalty.weight = -5.0  # strong penalty
        self.rewards.legstand_drift_penalty.weight = -0.5  # tiny penalty
        self.rewards.legstand_bonus_upright.weight = 5.0  # sparse bonus
        # ----------------------------------------

        # terrain settings
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeGo2LegStandReworkedEnvCfg_PLAY(UnitreeGo2LegStandReworkedEnvCfg):

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
