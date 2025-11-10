from isaaclab.utils import configclass
#from isaaclab.envs.go2_env_cfg import UnitreeGo2RoughEnvCfg
#from isaaclab.envs.rsl_rl import RslRlOnPolicyRunnerCfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from .source.isaaclab_tasks.isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg



@configclass
class Go2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Env setup
        self.scene.num_envs = 2048


        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None # Remove terrain curriculum
        

        # Observation dimension padding (if 243-dim needed)
        self.observations.policy.num_observations = 243

        # Control
        self.control.stiffness = {"joint": 25.0}
        self.control.damping = {"joint": 0.5}
        self.control.action_scale = 0.25

        # Rewards
        self.rewards.tracking_lin_vel.weight = 1.5
        self.rewards.lin_vel_z.weight = -2.0
        self.rewards.ang_vel_xy.weight = -0.05
        self.rewards.tracking_ang_vel.weight = 0.75
        self.rewards.torques.weight = -0.0002
        self.rewards.dof_acc.weight = -2.5e-7
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.action_rate.weight = -0.01

        # Init state
        self.init_state.pos = [0.0, 0.0, 0.42]
        self.init_state.default_joint_angles = {
            "FL_hip_joint": 0.1, "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8, "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5, "RR_calf_joint": -1.5,
        }

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25



        
@configclass
class Go2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    # PPO hyperparameters
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 200
    experiment_name = "go2_flat_custom"
    run_name = "exp1"
