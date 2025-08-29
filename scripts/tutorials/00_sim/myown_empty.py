

# first launch Isaac Sim:

import argparse
from isaaclab.app import AppLauncher

# add args
parser = argparse.ArgumentParser(description="My own empty stage")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app:
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# note: import only after sim app is running



from isaaclab.sim import SimulationCfg, SimulationContext 


def main():
    """Main Function to setup and loop simulation"""

    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0,0.0,0.0])
    
    sim.reset()
    print("Info: Setup Complete...")

    while simulation_app.is_running():
        # loop
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()
