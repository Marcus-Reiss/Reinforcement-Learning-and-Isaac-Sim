# Aim of this script: 
# - Spawn cuboids in a scene
# - Give velocity commands to the cuboids

import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Trying to give velocity commands to cuboids")

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
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg
from omni.isaac.lab.sim import SimulationContext

def setup_scene():

    # Ground and lights ----------------------------------------------------

    # Prim paths
    prim_path_gp = "/World/defaultGroundPlane"
    prim_path_li = "/World/Light"

    # Ground Plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func(prim_path_gp, cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func(prim_path_li, cfg)

    # Cuboids --------------------------------------------------------------

    # Creating an array of origins
    origins = [[0.25, -0.5, 0.], [0.25, 0.5, 0.]]

    # Creating prims for the origins
    for k, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{k + 1}", "Xform", translation=origin)

    # Adding the deformable cuboids
    cuboid_cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
        size=(0.5, 0.5, 0.5),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.9)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            rot=torch.tensor([0.0, 0.0, 0.0, 1.0])
        ),
        debug_vis=True,
    )
    cuboid_object = DeformableObject(cfg=cuboid_cfg)

    return origins, cuboid_object

def run_simulation(sim: sim_utils.SimulationContext, origins: torch.Tensor, cuboid_object: DeformableObject):

    # Getting time increment, initializing sim time and count variable
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Starting simulation
    while simulation_app.is_running():

        # Resetting simulation after N steps
        if count % 300 == 0:

            # 1. Getting the default state for each cuboid object
            nodal_state = cuboid_object.data.default_nodal_state_w.clone()

            # 2. Setting the cuboids to their respective origins
            # pos_w = torch.tensor(origins, device=sim.device)
            # nodal_state[..., :3] = cuboid_object.transform_nodal_pos(nodal_state[..., :3], pos_w)

            # 3. Writing nodal state to simulation
            cuboid_object.write_nodal_state_to_sim(nodal_state)

            # 4. Reset buffers
            cuboid_object.reset()

        cuboid_object.update(sim_dt)

        # Perform step
        sim.step()
        sim_time += sim_dt
        count += 1


def main():

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])

    # Design scene
    scene_origins, scene_cuboids = setup_scene()

    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulation(sim, scene_origins, scene_cuboids)


if __name__ == "__main__":

    # running the main function
    main()

    # closing simulation app
    simulation_app.close()
