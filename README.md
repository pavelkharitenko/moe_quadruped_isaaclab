
## Multitask Reinforcement Learning of Quadrupeds based on Mixtures of Experts and Bayesian Nonparametric Representations

![](figures/framework_overview.png)



## Installation


### Install IsaacGym, IsaacLab & clone this repository

To install this code, follow the official [IsaacLab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) but clone this repository instead of the default repo in their docs. 

(Notice: This repository uses the __bnpy__ python library for the DPMM implementation, which is not available on Windows. Install and run on Linux or don't use the DPMM related code. The multitask environment can be still run on Windows but DPMM-VAE is not available then.)

1. [Install IsaacSim (Choose pip or binary installation, what you prefer.)](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) 

 - When runnning `pip install "isaacsim[all,extscache]==4.5.0"` and `conda create -n env_isaaclab python=3.10`, we used IsaacLab __4.5.0__ and python __3.10__, but newer versions may work as well.

 - When running `pip install -U torch== ...` install the pytorch version suitable for your GPU. We used PyTorch version __2.8.0+cu126__ and TorchVision version __0.23.0+cu126__. Use `pip install --force-reistall torch torchvision ... pytorch.org/whl/cu12x (lower x)` if V100 GPU is outdated for your PyTorch version. After downgrading, numpy might be too new, try to downgrade via `pip install numpy<2`

2. [Install IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab): 

- Clone __this__ repository.

- When running `isaaclab.sh --install`, run without arguments to install all RL libraries or add `rsl_rl`. This repo uses RSL RL for the algorithms (installing skrl, sb3, etc. not needed).


### Installing python libraries

1. Install missing python packages for pip, inside the conda environment created in the last section.

- Run the single task as a test to see if your isaaclab installation and python packages have been installed completely:


```bash
conda activate env_isaaclab


cd moe_quadruped_isaaclab/

python scripts/moe_quadruped/train_single.py --task=Isaac-Velocity-Flat-Unitree-Go2-MoE-v0  --num_envs=256  --max_iterations=200 --experiment_name=flat_go2_single --run_name=run_flatveltracking --headless
```
 
This command might take a while to run, you should see rewards being logged and iteration numbers after a while.

2. Install Visualization tools

`cd vis_utils/` 

and then run 

`python -m pip install -e .`


3. Other packages:

- We provide a requirements.txt and environment.yml file for reference, but we recommend installing missing packages at hand. Some versions might become uncompatible with newer isaaclab. 

- The PyTorch library might consider your GPU as outdated. Try to downgrade Pytorch like written above. Also might be necesary to downgrade `numpy` afterwards. Try then to downgrade your numpy like written above some subversions down until no error is available.


## Commands


### Running the Multitask Environment

Recommended to use 2048 parallel envs and 2-5K timesteps. Some tasks may converge before 2K timesteps.

The arg `--num_envs=2048` is per task. Using e.g. 4 tasks, a total of 8192 envs will be created.

Add `--headless` flag when runnning on a remote server. Add `--append_task_id` if one-hot vector needed.

### Run MT4-Env (four quadruped tasks) with PPO via 

```
python scripts/moe_quadruped/train_multitask_dpmm_conditioned.py --task=Isaac-MT-Unitree-Go2-v0 --num_envs=2048 --max_iterations=5000  --run_name=run_multitask_ppo_final_52 --seed 52 --append_task_id --headless
```
### Run MT4-Env with MoE-PPO architecture via:

```
python scripts\moe_quadruped\train_multitask_moe.py --task=Isaac-MT-Unitree-Go2-v0 --num_envs=2048 --max_iterations=5000  --run_name=run_multitask_final_moe_52  --seed 52 --append_task_id --headless
```


### Run MT4-Env with DPMM-VAE PPO via 


```
python scripts/moe_quadruped/train_multitask_dpmm_bnpy_ppo.py --task=Isaac-MT-Unitree-Go2-v0 --num_envs=512 --max_iterations=5000  --run_name=run_multitask_final_dpmm_vae_ppo_449  --seed 449 --append_task_id --headless 
```



### Single task environments

Run singe task training using the train_single_ppo or train_single_moe scripts with the commands as follows

Run Flat Velocity Tracking Task training:

```
python scripts/moe_quadruped/train_single.py --task=Isaac-Velocity-Flat-Unitree-Go2-MoE-v0  --num_envs=4096  --max_iterations=2000 --experiment_name=flat_go2_single --run_name=run_flatveltracking --headless
```

Run Handstand Task training:

```
python scripts/moe_quadruped/train_single_ppo.py --task=Isaac-Velocity-Flat-HandStand-Unitree-Go2-v0 --num_envs=2048 --max_iterations=1500 --experiment_name=handstand_go2_st --run_name=run_handstand --headless 
```

Run Legstand Task training:

```
python scripts/moe_quadruped/train_single_ppo.py --task=Isaac-Velocity-LegStand-Unitree-Go2-MoE-v0 --num_envs=2048 --max_iterations=1500 --experiment_name=legstand_go2_st --run_name=run_legstand --headless
```


Run Crawl Task:

```
python scripts/moe_quadruped/train_single_ppo.py --task=Isaac-Velocity-Lowcrawl-Unitree-Go2-MoE-v0 --num_envs=2048 --max_iterations=1500 --experiment_name=lowcrawl_go2_st --run_name=run_lowcrawl --headless
```




## Multitask Environment

Code based on original author of MT-IsaacLab https://github.com/meenalparakh/MT-IsaacLab.

### Add new tasks:

1. Create singletask EnvCfg class in `moe_quadruped_isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2_moe/`

2. If everything works fine in singletask, add your EnvCfg class to `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion_multitask/velocity/config/go2_mt/`

3. Your EnvCfg added in `go2_mt` must subclass a MT-Env, e.g. `go2_mt.flat_env_cfg.UnitreeGo2FlatEnvCfg` class. RewardsCfg can still be imported from your single task, no need to write again.

4. Add your EnvCfg task as an attribute to `go2_mt/mt4_env_cfg.py` as `myEnvTask: TaskConfigs = UnitreeGo2MyEnvCfg()`



## Folder structure

### Tasks

Single tasks code in 

```
source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2_moe\
```

Multitask code in

```
source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion_multitask\velocity\config\go2_mt\
```

`mt4_env_cfg.py` implements the joint multitask environment.


### Scripts

All our training scripts `train_ ... .py` are in  

```
scripts\moe_quadruped\
```


Custom RSL RL code for `MoE Actor`, `DPMM-VAE` in

```
source\isaaclab_rl\isaaclab_rl\rsl_rl\
```

and 


```
source\isaaclab_rl\isaaclab_rl\rsl_rl\DPMM\
```


For DPMM-VAE, Trainer code located in

```
source\isaaclab_rl\isaaclab_rl\rsl_rl\DPMM\MELTS\tigr\trainer\dpmm_trainer.py
```

DPMM-VAE additional methods, like buffer code and sampling located in 

```
source\isaaclab_rl\isaaclab_rl\rsl_rl\DPMM\dpmm_utils.py
```

