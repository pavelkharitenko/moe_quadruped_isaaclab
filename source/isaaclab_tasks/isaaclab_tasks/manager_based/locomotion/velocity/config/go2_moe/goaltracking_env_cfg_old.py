# goaltracking_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass
from .flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from .goaltracking_rewards import *
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class UnitreeGo2GoalTrackingRewardsCfg(RewardsCfg):
    """Custom reward terms for goal-tracking task."""

    distance = RewTerm(
        func=goal_distance,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    heading = RewTerm(
        func=goal_heading,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    success = RewTerm(
        func=goal_success,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"])},
    )

    progress = RewTerm(
        func=goal_progress,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    orientation = RewTerm(
        func=base_orientation_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    base_height = RewTerm(
        func=base_height_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CommandsCfg:
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0,8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )



@configclass
class UnitreeGo2GoalTrackingEnvCfg(UnitreeGo2FlatEnvCfg):
    """Environment config for goal-tracking with UnitreeGo2."""

    rewards: UnitreeGo2GoalTrackingRewardsCfg = UnitreeGo2GoalTrackingRewardsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable velocity command
        #self.commands.base_velocity = None
        # disable old velocity command
        self.commands.base_velocity = None

        # replace observation term to expose new goal_position instead
        self.observations.policy.
        commands.params["command_name"] = "goal_position"

        # disable default locomotion rewards
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0

        # set weights for goal-tracking
        self.rewards.distance.weight = 2.0
        self.rewards.heading.weight = 1.0
        self.rewards.success.weight = 10.0
        self.rewards.progress.weight = 0.5
        self.rewards.orientation.weight = -2.0
        self.rewards.base_height.weight = -1.0
        #self.rewards.feet_air_time.weight = 0.5

        

        


class UnitreeGo2GoalTrackingEnvCfg_PLAY(UnitreeGo2GoalTrackingEnvCfg):
    """Play mode: fewer envs, no randomization, easier debugging."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization
        self.observations.policy.enable_corruption = False
        # remove random pushes
        self.events.base_external_force_torque = None
        self.events.push_robot = None


import math
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.navigation.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# base locomotion env for Unitree
from isaaclab_tasks.manager_based.locomotion.velocity.config.unitree_go2.rough_env_cfg import UnitreeGo2RoughEnvCfg


LOW_LEVEL_ENV_CFG = UnitreeGo2RoughEnvCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5, params={"std": 2.0, "command_name": "pose_command"}
    )
    position_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5, params={"std": 0.2, "command_name": "pose_command"}
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2, params={"command_name": "pose_command"}
    )


@configclass
class CommandsCfg:
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class UnitreeGo2GoalTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Navigation environment for Unitree Go2 tracking a goal pose."""

    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene.replace(robot=UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"))
    actions = LOW_LEVEL_ENV_CFG.actions
    observations = ObservationsCfg()
    events = EventCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
