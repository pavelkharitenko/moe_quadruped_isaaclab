# lowcrawl_env_cfg.py

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm

from .flat_env_cfg import UnitreeGo2FlatEnvCfg  # import MT FlatEnv config to subclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_moe.lowcrawl_env_cfg import UnitreeGo2LowCrawlRewardsCfg  # still use original singletask handstand env rewards


@configclass
class UnitreeGo2LowCrawlEnvCfg(UnitreeGo2FlatEnvCfg):

    rewards: UnitreeGo2LowCrawlRewardsCfg = UnitreeGo2LowCrawlRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # disable incompatible rewards
        self.rewards.feet_air_time.weight = 0.125  # default hopping reward
        self.rewards.flat_orientation_l2.weight = 0.0

        # disable reset on base touching the ground
        #self.terminations.base_contact = None

        # velocity tracking
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.25

        # low-crawl specific shaping
        self.rewards.base_height.weight = 0.2
        self.rewards.feet_clearance.weight = 0.1
        self.rewards.feet_contact_fraction.weight = 0.05  #1.0
        self.rewards.posture_orientation.weight = 0.125

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Head_lower", "Head_upper", "base"]
        self.terminations.base_contact.time_out = True


class UnitreeGo2LowCrawlEnvCfg_PLAY(UnitreeGo2LowCrawlEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
