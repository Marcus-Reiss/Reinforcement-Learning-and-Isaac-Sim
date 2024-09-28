import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")

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
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext


def setup_scene():
    """ Include here all code for setup the objects """
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()

    # Spawn the object:
    prim_path = "/World/defaultGroundPlane"
    sim_utils.spawn_from_usd(prim_path, cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3", "Origin4"
    origins = [[0.25, 0.25, 0.],[-.25, 0.25, 0.], [0.25, -.25, 0.], [-.25, -.25, 0.]]
    prim_utils.create_prim(f"/World/Origin1", "Xform", translation=[0.25, 0.25, 0.])
    prim_utils.create_prim(f"/World/Origin2", "Xform", translation=[-.25, 0.25, 0.])
    prim_utils.create_prim(f"/World/Origin3", "Xform", translation=[0.25, -.25, 0.])
    prim_utils.create_prim(f"/World/Origin4", "Xform", translation=[-.25, -.25, 0.])

    # Rigid Object
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone_object = RigidObject(cfg=cone_cfg)

    return cone_object, origins


def run_simulation(sim: sim_utils.SimulationContext, entities: RigidObject, origins: torch.Tensor):
    """ Include here all code for running the simulation """
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 1. Get the default state for each object
    root_state = entities.data.default_root_state.clone()

    # 2. Change the position of cylinders for matching each origin
    root_state[:, :3] += torch.tensor(origins)

    # 3. Write root state to simulation
    entities.write_root_state_to_sim(root_state)

    # 4. Reset buffers
    entities.reset()

    while simulation_app.is_running():
    # TODO HERE
        # perform step
        sim.step()
        sim_time += sim_dt
        count += 1

        # update buffers
        entities.update(sim_dt)

        if count == 250:
            


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