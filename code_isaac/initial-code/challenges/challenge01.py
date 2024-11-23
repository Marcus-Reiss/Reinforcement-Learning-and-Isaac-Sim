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
    prim_path = "/World/defaultGroundPlane"

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func(prim_path,cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3", "Origin4"
    origins = [[0.25, 0.25, 0.],[-.25, 0.25, 0.], [0.25, -.25, 0.], [-.25, -.25, 0.]]
    # prim_utils.create_prim(f"/World/Origin1", "Xform", translation=[0.25, 0.25, 0.])
    # prim_utils.create_prim(f"/World/Origin2", "Xform", translation=[-.25, 0.25, 0.])
    # prim_utils.create_prim(f"/World/Origin3", "Xform", translation=[0.25, -.25, 0.])
    # prim_utils.create_prim(f"/World/Origin4", "Xform", translation=[-.25, -.25, 0.])
    for k, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{k + 1}", "Xform", translation=origin)

    # Rigid Object
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origin[1-3]/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.2,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone_object = RigidObject(cfg=cone_cfg)

    # Adding a blue deformable cuboid
    cuboid_cfg = DeformableObjectCfg(
        prim_path="/World/Origin4/Cube",
        spawn=sim_utils.MeshCuboidCfg(
        size=(0.5, 0.5, 0.5),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(),
        debug_vis=True,
    )
    # cfg_cuboid_deformable.func("/World/Origin4/CuboidDeformable", cfg_cuboid_deformable, translation=origins[3])
    cuboid_object = DeformableObject(cfg=cuboid_cfg)

    return cone_object, cuboid_object, origins


def run_simulation(sim: sim_utils.SimulationContext, entities: RigidObject, cuboid: DeformableObject, origins: torch.Tensor):
    """ Include here all code for running the simulation """
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        if count % 250 == 0:
            
            # 1. Get the default state for each object (cones)
            root_state = entities.data.default_root_state.clone()

            # 1.2. Get the default state for cuboid object
            nodal_state = cuboid.data.default_nodal_state_w.clone()

            # 2. Change the position of cones for matching each origin
            root_state[:, :3] += torch.tensor(origins[:3]) + math_utils.sample_cylinder(
            radius=0.1, h_range=(3.0, 3.5),
            size=entities.num_instances,
            device=entities.device
            )

            # Trying to do the same for the cuboid
            # pos_w = torch.rand(cuboid.num_instances, 3, device=sim.device)*0.8 + origins[3]
            pos_w = torch.tensor(origins[3], device=sim.device)

            random_offset = math_utils.sample_cylinder(
            radius=0.2, h_range=(2.0, 2.5),
            size=cuboid.num_instances,
            device=sim.device
            )

            pos_w += random_offset[0, :3]
            # quat_w = math_utils.random_orientation(cuboid.num_instances, device=sim.device)
            nodal_state[..., :3] = cuboid.transform_nodal_pos(nodal_state[..., :3], pos_w)

            # 3. Write root state to simulation
            entities.write_root_state_to_sim(root_state)
            cuboid.write_nodal_state_to_sim(nodal_state)

            # 4. Reset buffers
            entities.reset()
            cuboid.reset()


        entities.update(sim_dt)

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
    scene_entities, scene_cuboid, scene_origins = setup_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulation(sim, scene_entities, scene_cuboid, scene_origins)

if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()