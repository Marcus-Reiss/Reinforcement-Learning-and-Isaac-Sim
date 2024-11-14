import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulated object")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab_assets import CARTPOLE_CFG
from omni.isaac.lab.sim import SimulationContext


def setup_scene():
    
    cfg = sim_utils.GroundPlaneCfg()

    prim_path = "/World/defaultGroundPlane"
    sim_utils.spawn_from_usd(prim_path, cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    origins = [[0.25, 0.25, 0.],[-.25, 0.25, 0.], [0., -.25, 0.]]
    prim_utils.create_prim(f"/World/Origin1", "Xform", translation=[0.25, 0.25, 0.])
    prim_utils.create_prim(f"/World/Origin2", "Xform", translation=[-.25, 0.25, 0.])
    prim_utils.create_prim(f"/World/Origin3", "Xform", translation=[0.25, -.25, 0.])

    # Articulated Object (Cartpole)
    cartpole_cfg = CARTPOLE_CFG.replace(prim_path="/World/Origin.*/Robot")
    cartpole = Articulation(cfg=cartpole_cfg)

    return cartpole_object, origins

def run_simulation(sim: sim_utils.SimulationContext, entities: CARTPOLE_CFG, origins: torch.Tensor):
    """ Include here all code for running the simulation """
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        # set joint positions and velocities with some noise
        joint_pos, joint_vel = entities.data.default_joint_pos.clone(),
        entities.data.default_joint_vel.clone()
        joint_pos += torch.rand_like(joint_pos) * 0.1
        entities.write_joint_state_to_sim(joint_pos, joint_vel)

        # Apply random action
        # 1. generate random joint efforts
        efforts = torch.randn_like(entities.data.joint_pos) * 5.0
        # 2. apply action to the robot
        entities.set_joint_effort_target(efforts)
        # 3. write data to sim
        entities.write_data_to_sim()
        entities.update(sim_dt)

    sim.step()
    sim_time += sim_dt
    count += 1


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    
    # Design scene
    scene_entities, scene_origins = setup_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulation(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
