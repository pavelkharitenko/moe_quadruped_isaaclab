# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import math

from .flat_env_cfg import UnitreeGo2FlatEnvCfg  # import MT FlatEnv Config to subclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_moe.legstand_env_cfg import UnitreeGo2LegStandRewardsCfg  # import rewards of original singletask legstand


@configclass
class UnitreeGo2LegStandEnvCfg(UnitreeGo2FlatEnvCfg):

    # add custom rewards
    rewards: UnitreeGo2LegStandRewardsCfg = UnitreeGo2LegStandRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards, no vel. tracking in legstand
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0

        # legstand-specific reward
        self.rewards.front_feet_height.weight = 1.25
        self.rewards.orientation.weight = -2.5
        self.rewards.front_air.weight = 1.5
        self.rewards.back_support.weight = 1.0
        self.rewards.base_penalty.weight = 2.5

        #self.rewards.feet_air_time.weight = 5.0
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        self.terminations.base_contact.time_out = True


class UnitreeGo2LegStandEnvCfg_PLAY(UnitreeGo2LegStandEnvCfg):

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
