# Aim of this script: 
# - Spawn cuboids in a scene
# - Give velocity commands to the cuboids

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
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from omni.isaac.lab.sim import SimulationContext


def setup_scene():
    """ Include here all code for setup the objects """

    # Prim-path
    prim_path_gp = "/World/defaultGroundPlane"
    prim_path_li = "/World/Light"

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func(prim_path_gp,cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func(prim_path_li, cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3", "Origin4"
    origins = [[0.0, -1.0, 0.], [0.0, 1.0, 0.], [-21.0, 0., 0.], [-20.0, 0., 0.], [-23.0, 0., 0.], [-26.0, 0., 0.], [-29.0, 0., 0.]]
    for k, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{k + 1}", "Xform", translation=origin)

    # Rigid Object (two cuboids that will move)
    cuboid_cfg = RigidObjectCfg(
        prim_path="/World/Origin[1-2]/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=[0.5, 0.5, 0.5],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.7, 0.7), roughness=0.8, metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cuboid_object = RigidObject(cfg=cuboid_cfg)

    # Rigid Object (a cuboid that will be stationary)
    cuboid_cfg2 = RigidObjectCfg(
        prim_path="/World/Origin[4-7]/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=[0.5, 2.5, 5.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0), roughness=0.8, metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cuboid_object2 = RigidObject(cfg=cuboid_cfg2)

    # Rigid Object (a cuboid that will be stationary)
    cuboid_cfg3 = RigidObjectCfg(
        prim_path="/World/Origin3/Cuboid",
        spawn=sim_utils.CuboidCfg(
            size=[0.5, 2.5, 0.3],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=0.8, metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cuboid_object3 = RigidObject(cfg=cuboid_cfg3)

    return cuboid_object, origins


def velocity_values():

    # Linear velocitiy
    v = [-2.5, 0.0, 0.0]

    # Angular velocity
    w = [0.0, 0.0, 7.0]

    # Extended list
    vw = v
    vw.extend(w)

    return vw


def run_simulation(sim: sim_utils.SimulationContext, entities: RigidObject, origins: torch.Tensor):
    """ Include here all code for running the simulation """
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Defining velocity commands for the two cuboids
    vw = velocity_values()
    vel = torch.tensor([vw, vw], device=sim.device)

    while simulation_app.is_running():

        if count % 800 == 0:
            
            # 1. Get the default state for each object (cones)
            root_state = entities.data.default_root_state.clone()

        #     # 2. Change the position of cones for matching each origin
        #     root_state[:, :3] += torch.tensor(origins[:3]) + math_utils.sample_cylinder(
        #     radius=0.1, h_range=(3.0, 3.5),
        #     size=entities.num_instances,
        #     device=entities.device
        #     )

            # 3. Write root state to simulation
            entities.write_root_state_to_sim(root_state)

            # 4. Reset buffers
            entities.reset()

        entities.update(sim_dt)

        # Applying velocities every dt (every simulation's frame)
        entities.write_root_velocity_to_sim(root_velocity=vel)

        # perform step
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
