# lowcrawl_env_cfg.py

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import math

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from .lowcrawl_rewards import *


@configclass
class UnitreeGo2LowCrawlRewardsCfg(RewardsCfg):
    """Rewards for low-profile crawling with velocity tracking."""

    # --- Core crawling posture rewards ---

    base_height = RewTerm(
        func=lowcrawl_base_height_signed,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "target_height": 0.18,
            "tolerance": 0.15,
        },
    )

    posture_orientation = RewTerm(
        func=lowcrawl_orientation_signed,
        weight=0.0,
        params={
            "max_tilt_rad": 0.35,  # ~20 degrees
        },
    )


    feet_clearance = RewTerm(
        func=lowcrawl_feet_clearance_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            ),
            "target_height": 0.06,
            "std": math.sqrt(0.0025),
        },
    )

    feet_contact_fraction = RewTerm(
        func=lowcrawl_feet_contact_fraction,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            ),
        },
    )

    feet_air_time = RewTerm(
        func=lowcrawl_feet_last_air_time_reward,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            ),
            "max_air_time": 0.15,
        },
    )





from .flat_env_cfg import UnitreeGo2FlatEnvCfg


@configclass
class UnitreeGo2LowCrawlEnvCfg(UnitreeGo2FlatEnvCfg):

    rewards: UnitreeGo2LowCrawlRewardsCfg = UnitreeGo2LowCrawlRewardsCfg()

    def __post_init__(self):
        super().__post_init__()


        # disable incompatible rewards
        self.rewards.feet_air_time.weight = 0.0  # default hopping reward
        self.rewards.flat_orientation_l2.weight = 0.0

        # disable reset on base touching the ground
        #self.terminations.base_contact = None

        # velocity tracking
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.5

        # low-crawl specific shaping
        self.rewards.base_height.weight = 0.5
        self.rewards.feet_clearance.weight = 0.4
        self.rewards.feet_contact_fraction.weight = 1.0
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.posture_orientation.weight = 0.5


        # terrain & observations
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


class UnitreeGo2LowCrawlEnvCfg_PLAY(UnitreeGo2LowCrawlEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
