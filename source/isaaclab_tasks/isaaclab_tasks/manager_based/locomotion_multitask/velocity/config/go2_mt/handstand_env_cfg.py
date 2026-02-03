# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import math

from .flat_env_cfg import UnitreeGo2FlatEnvCfg  # import MT FlatEnv config to subclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_moe.handstand_env_cfg import UnitreeGo2HandStandRewardsCfg  # still use original singletask handstand env rewards


@configclass
class UnitreeGo2HandStandEnvCfg(UnitreeGo2FlatEnvCfg):

    # add custom rewards
    rewards: UnitreeGo2HandStandRewardsCfg = UnitreeGo2HandStandRewardsCfg()

    #base_link_name = "base"
    #foot_link_name = ".*_foot"
    # fmt: off
    #joint_names = [
    #    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    #    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    #]

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.flat_orientation_l2.weight = 0

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.05

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]  #[self.foot_link_name]

        # HandStand
        handstand_type = "back"  # which leg on air, can be "front", "back", "left", "right"
        if handstand_type == "front":
            air_foot_name = "F.*_foot"
            self.rewards.handstand_orientation_l2.weight = -1.0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [-1.0, 0.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.6
        elif handstand_type == "back":
            air_foot_name = "R.*_foot"
            self.rewards.handstand_orientation_l2.weight = -1.0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [1.0, 0.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.6
        elif handstand_type == "left":
            air_foot_name = ".*L_foot"
            self.rewards.handstand_orientation_l2.weight = 0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [0.0, -1.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.3
        elif handstand_type == "right":
            air_foot_name = ".*R_foot"
            self.rewards.handstand_orientation_l2.weight = 0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [0.0, 1.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.3

        self.rewards.handstand_feet_height_exp.weight = 2.5
        self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_on_air.weight = 1.0
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_air_time.weight = 1.0
        self.rewards.handstand_feet_air_time.params["sensor_cfg"].body_names = [air_foot_name]

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        self.terminations.base_contact.time_out = True
