# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedMTRLEnvCfg, TaskConfigs
from isaaclab.utils import configclass

# import single task envs
from .flat_env_cfg import UnitreeGo2FlatEnvCfg
from .rough_env_cfg import UnitreeGo2RoughEnvCfg
from .legstand_env_cfg import UnitreeGo2LegStandEnvCfg


@configclass
class MTLocomotionEnvCfg(ManagerBasedMTRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    flatVel: TaskConfigs = UnitreeGo2FlatEnvCfg()
    legStand: TaskConfigs = UnitreeGo2LegStandEnvCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.num_multi_task_envs = 2
        self.task_spacing = 2.5
        self.num_envs_per_task = 128
        self.envs_spacing = 2.5
        self.append_task_id = False
        self.concatenate_step_results = True

        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005

        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        
        # TODO from singletask LocomotionVelocityRoughEnvCfg add post_init settings here (to task RoughEnv)

        self.sim.physics_material = self.flatVel.scene.terrain.physics_material # here scene.terrain is red marked
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.flatVel.scene.height_scanner is not None:
            self.flatVel.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.flatVel.scene.contact_forces is not None:
            self.flatVel.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.flatVel.curriculum, "terrain_levels", None) is not None:
            if self.flatVel.scene.terrain.terrain_generator is not None:
                self.flatVel.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.flatVel.scene.terrain.terrain_generator is not None:
                self.flatVel.scene.terrain.terrain_generator.curriculum = False
        
    




class MTLocomotionEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
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

       