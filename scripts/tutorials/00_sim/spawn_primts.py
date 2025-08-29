

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

import isaacsim.core.utils.prims as prim_utils

from isaaclab.sim import SimulationCfg, SimulationContext 

import isaaclab.sim as sim_utils

def create_prims():

    # ground
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")

    # cone
    cfg_cone = sim_utils.ConeCfg(
        radius=0.14,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0))
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))


    # light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

def main():
    """Main Function to setup and loop simulation"""

    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # add prims to scene
    create_prims()

    
    sim.reset()
    print("Info: Setup Complete...")

    while simulation_app.is_running():
        # loop
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()
