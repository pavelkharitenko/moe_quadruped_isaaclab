# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import TaskConfigs
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs.utils.mtrl import task_id_onehot


import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg,
    ActionsCfg,
    ObservationsCfg,
    CurriculumCfg,
    EventCfg,
    RewardsCfg,
    TerminationsCfg
)


##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# MT Observation
##

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, ObsTerm



@configclass
class LocomotionVelocityObservationsCfg(ObservationsCfg):

    # Policy group inherits everything from single-task
    @configclass
    class LocomotionVelocityPolicyCfg(ObservationsCfg.PolicyCfg):
        # keep existing velocity, gravity, etc.

        # MT Observation: task ID as one-hot
        # assume your env has a property `task_id` which is an int from 0..num_tasks-1
        task_id = ObsTerm(
            func=task_id_onehot,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: LocomotionVelocityPolicyCfg = LocomotionVelocityPolicyCfg()




##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ground terrain

        #self.terrain = None
        
        self.world_terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
            
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        

        # robots
        self.robot: ArticulationCfg = MISSING

        # sensors
        self.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        self.contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
        
        # lights # TODO create self.world_sky_light to just have one in world, then see if same for self.terrain can be done
        self.world_sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

    def initialize(self):

        
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        



##
# Environment configuration
##
@configclass
class LocomotionVelocityRoughEnvCfg(TaskConfigs):
    """Configuration for the locomotion velocity-tracking environment."""

    episode_length_s: float = 20.0
    # Scene settings
    scene: MySceneCfg = MySceneCfg()
    #scene: MySceneCfg = MISSING
    # Basic settings
    observations: LocomotionVelocityObservationsCfg = LocomotionVelocityObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
