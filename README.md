![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows, such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks. Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.


## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:
- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Isaac Sim Version Dependency

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each release of Isaac Lab.
Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding dependency versions for Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version |
| ----------------------------- | ----------------- |
| `main` branch                 | Isaac Sim 4.5     |
| `v2.1.0`                      | Isaac Sim 4.5     |
| `v2.0.2`                      | Isaac Sim 4.5     |
| `v2.0.1`                      | Isaac Sim 4.5     |
| `v2.0.0`                      | Isaac Sim 4.5     |
| `feature/isaacsim_5_0` branch | Isaac Sim 5.0     |

Note that the `feature/isaacsim_5_0` will contain active updates and may contain some breaking changes
until the official Isaac Lab 2.2 release.
It currently requires the [Isaac Sim 5.0 branch](https://github.com/isaac-sim/IsaacSim) available on GitHub built from source.
Please refer to the README in the `feature/isaacsim_5_0` branch for instructions for using Isaac Lab with Isaac Sim 5.0.
We are actively working on introducing backwards compatibility support for Isaac Sim 4.5 for this branch.


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area in the
`Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## Connect with the NVIDIA Omniverse Community

Do you have a project or resource you'd like to share more widely? We'd love to hear from you!
Reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to explore opportunities
to spotlight your work.

You can also join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to
connect with other developers, share your projects, and help grow a vibrant, collaborative ecosystem
where creativity and technology intersect. Your contributions can make a meaningful impact on the Isaac Lab community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
## Commands

### Locally (Windows)


Train Flat Task
```
C:/Users/Pavel/miniconda3/envs/env_isaaclab/python.exe C:\Users\Pavel\IsaacLab\scripts\moe_quadruped\train_single.py --task=Isaac-Velocity-Flat-Unitree-Go2-MoE-v0  --num_envs=32  --max_iterations=50 --experiment_name=flat_go2_single --run_name=run17 --headless
```

Play Flat Task

```

```


Train Bipedal Task
```
C:/Users/Pavel/miniconda3/envs/env_isaaclab/python.exe C:\Users\Pavel\IsaacLab\scripts\moe_quadruped\train_moe.py --task=Isaac-Bipedal-Unitree-Go2-MoE-v0  --num_envs=1024  --max_iterations=2000 --experiment_name=bipedal_go2_single --run_name=run30_st_bipedal               
```

Play Bipedal Task & Record
```
C:/Users/Pavel/miniconda3/envs/env_isaaclab/python.exe C:\Users\Pavel\IsaacLab\scripts\moe_quadruped\play_moe.py --task=Isaac-Bipedal-Unitree-Go2-Play-MoE-v0   --load_run=2025-11-06_10-26-36_run31_st_bipedal --video --video_length 200 --num_envs=12
```



Train task with custom RSL_RL train script:

```
C:/Users/Pavel/miniconda3/envs/env_isaaclab/python.exe C:\Users\Pavel\IsaacLab\scripts\moe_quadruped\train_single.py --task=Isaac-Velocity-Flat-Unitree-Go2-MoE-v0  --num_envs=32  --max_iterations=50 --experiment_name=flat_go2_single --run_name=run17 --headless
```

Train in Multitask env:
```
C:/Users/Pavel/miniconda3/envs/env_isaaclab/python.exe C:\Users\Pavel\IsaacLab\scripts\moe_quadruped\train_multitask.py --task=Isaac-MT-Unitree-Go2-v0  --num_envs=38  --max_iterations=50 --experiment_name=flat_go2_mt --run_name=run19_mt
```


### Run Multitasks Envs

Run MT4-Env via 

```
python scripts/moe_quadruped/train_multitask_ppo.py --task=Isaac-MT-Unitree-Go2-v0 --num_envs=1024 --max_iterations=2500  --run_name=run_multitask_exp_ppo --headless --seed 52
```

Run diagnostics

```
C:/Users/Pavel/miniconda3/envs/env_isaaclab/python.exe .\scripts\moe_quadruped\plot_diagnostics.py --single .\logs\rsl_rl\unitree_go2_flat\2025-11-03_10-21-04_run28_st\ --multi .\logs\rsl_rl\unitree_go2_flat\2025-11-03_09-43-13_run26\
```


### Remote (Ubuntu)

Run singe task training using the train_single_ppo or train_single_moe scripts with the commands as follows

Run Flat Velocity Tracking Task training:

```
python scripts/moe_quadruped/train_single.py --task=Isaac-Velocity-Flat-Unitree-Go2-MoE-v0  --num_envs=4096  --max_iterations=2000 --experiment_name=flat_go2_single --run_name=run_flatveltracking --headless
```

Run Handstand Task training:

´´´
python scripts/moe_quadruped/train_single_ppo.py --task=Isaac-Velocity-Flat-HandStand-Unitree-Go2-v0 --num_envs=2048 --max_iterations=1500 --experiment_name=handstand_go2_st --run_name=run_handstand --headless 
´´´

Run Legstand Task training:

´´´
python scripts/moe_quadruped/train_single_ppo.py --task=Velocity-LegStand-Unitree-Go2-MoE-v0 --num_envs=2048 --max_iterations=1500 --experiment_name=legstand_go2_st --run_name=run_legstand --headless
´´´

Run Crawl Task:

```
python scripts/moe_quadruped/train_single_ppo.py --task=Isaac-Velocity-Lowcrawl-Unitree-Go2-MoE-v0 --num_envs=2048 --max_iterations=1500 --experiment_name=lowcrawl_go2_st --run_name=run_lowcrawl --headless
```




## Multitask Environment

Code based on original author of MT-IsaacLab https://github.com/meenalparakh/MT-IsaacLab

### Add new tasks:

1. Create singletask EnvCfg class in `moe_quadruped_isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_moe/`

2. If everything works fine in singletask, add your EnvCfg class to `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion_multitask/velocity/config/go2_mt/`

3. Your EnvCfg added in `go2_mt` must subclass a MT-Env, e.g. `go2_mt.flat_env_cfg.UnitreeGo2FlatEnvCfg` class. RewardsCfg can still be imported from your single task, no need to write again.

4. Add your EnvCfg task as an attribute to `go2_mt/mt4_env_cfg.py` as `myEnvTask: TaskConfigs = UnitreeGo2MyEnvCfg()`






## MT4 was created by:


1. Create in source/isaaclab/envs/ two files, `manager_based_mt_rl_env(_cfg).py`, export their classes in `__init__.py`

2. Create folder in source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion_multitask/velocity, with similar structure to how the normal locomotion/velocity folder is.

3. In the normal velocity/velocity_env_cfg.py, we have MySceneCfg and LocomotionVelocityRoughEnvCfg, here we change both to fit Multitask setting: MySceneCfg should differntiate world- and task specific scene elements through self.world_ prefix. LocomotionVelocityRoughEnvCfg should have no __post_init__(self) method and subclass from TaskConfigs.

4. In velocity/config/ dir, create the multitask env go2_mt. Here we need to rewrite the original go2 configs rough_env_cfg.py, flat_env_cfg.py, ... etc. to match the multitask configs:

- import LocomotionVelocityRoughEnvCfg from isaaclab_tasks.manager_based.locomotion_multitask instead from locomotion.velocity

- subclass it to create UnitreeGo2RoughEnvCfg

- remove super.__init__(), and just write everything inside __post_init__(self)

5. Write the actual Multitask Environment: create a file, mt_env_cfg.py, in which a MTLocomotion class subclasses ManagerBasedMTRL. 

- Put all tasks coded according to step 4. as attributes. 

- Write a function __post_init__(self) in which all setups of __post__init() of LocomotionVelocityRoughEnvCfg should be done.