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
    
    append_task_id: bool = False # set via cli args, e.g. --append_task_id True, for consistent RL alg. as well
    flatVel: TaskConfigs = UnitreeGo2FlatEnvCfg()
    flatVel2: TaskConfigs = UnitreeGo2FlatEnvCfg()

    #legStand: TaskConfigs = UnitreeGo2LegStandEnvCfg()
    #legStand2: TaskConfigs = UnitreeGo2LegStandEnvCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.sim.dt = 0.005
        self.decimation = 4
        self.num_multi_task_envs = 2
        self.task_spacing = 20.0
        self.num_envs_per_task = 128
        self.envs_spacing = 2.5
        #self.append_task_id = True
        self.concatenate_step_results = True

        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods

        # TODO from singletask LocomotionVelocityRoughEnvCfg add post_init settings here (to task RoughEnv)

        # scale down the terrains because the robot is small

        self.flatSceneInits(self.flatVel.scene)
        self.flatSceneInits(self.flatVel2.scene)


        

    def flatSceneInits(self, scene):
        """No need in flatScene
        if scene.height_scanner is not None:
            scene.height_scanner.update_period = self.decimation * self.sim.dt
        """

        if scene.contact_forces is not None:
            scene.contact_forces.update_period = self.sim.dt
        """No need in flatScene
        scene.world_terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        scene.world_terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        scene.world_terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        """
        """No need in flatScene
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if scene.terrain.terrain_generator is not None:
                scene.terrain.terrain_generator.curriculum = True
        else:
            if scene.terrain.terrain_generator is not None:
                scene.terrain.terrain_generator.curriculum = False
        """

    


"""
class MTLocomotionEnvCfg_PLAY(MTLocomotionEnvCfg):
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
"""
