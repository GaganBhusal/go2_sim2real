import genesis as gs
import torch
from src.utils import JointController
import numpy as np


gs.init(backend=gs.gpu, logging_level = "warning")

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
go = scene.add_entity(
    gs.morphs.URDF(file='//home/yayy/My/Codeeeeee/Simulators/Genesis/genesis-world/genesis/assets/urdf/go2/urdf/go2.urdf'),
)

scene.build()


joints_local_idx = []
ranges = []
for links in go.links:
    joints = links.joints
    for joint in joints:
        joints_local_idx.append(joint.dof_idx_local)
        ranges.append(np.rad2deg(joint.dofs_limit))
        print(f"Joint Name : {joint.name}, DOF : {joint.n_dofs}, type : {type(joint)}, Axis : {torch.rad2deg(torch.tensor(joint.dofs_limit))}")


total_joints = 12

# tensor([[-60.0001,  60.0001]], dtype=torch.float64)
# tensor([[-60.0001,  60.0001]], dtype=torch.float64)
# tensor([[-60.0001,  60.0001]], dtype=torch.float64)
# tensor([[-60.0001,  60.0001]], dtype=torch.float64)
# tensor([[-90.0002, 200.0024]], dtype=torch.float64)
# tensor([[-90.0002, 200.0024]], dtype=torch.float64)
# tensor([[-30.0001, 260.0025]], dtype=torch.float64)
# tensor([[-30.0001, 260.0025]], dtype=torch.float64)
# tensor([[-155.9992,  -48.0001]], dtype=torch.float64)
# tensor([[-155.9992,  -48.0001]], dtype=torch.float64)
# tensor([[-155.9992,  -48.0001]], dtype=torch.float64)
# tensor([[-155.9992,  -48.0001]], dtype=torch.float64)


# joint_names = [FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint]

print(ranges)


go.set_dofs_kp(
    torch.tensor([100] * 12),
    dofs_idx_local = joints_local_idx[1:]

)

go.set_dofs_kv(
    torch.tensor([10] * 12),
    dofs_idx_local = joints_local_idx[1:]

)


for links in go.links[1:]:
    joints = links.joints
    for joint in joints:
        print(f"Joint name: {joint.name}, local index: {joint.dof_idx_local}")


controller = JointController(ranges[1:], [0, 0, 0, 0, 50, 50, 55, 55, -90, -90, -90, -90])

while True:
    z_coordinate_base = go.get_link("base").get_pos()[2]
    # print(z_coordinate_base)

    contact_forces = go.get_links_net_contact_force()

    print(f"The forces are :\nFL_hip_joint : {contact_forces[1]}\nFR_hip_joint : {contact_forces[2]}\nRL_hip_joint : {contact_forces[3]}\nRR_hip_joint : {contact_forces[4]}\nFL_thigh_joint : {contact_forces[5]}\nFR_thigh_joint : {contact_forces[6]}\nRL_thigh_joint : {contact_forces[7]}\nRR_thigh_joint : {contact_forces[8]}\nFL_calf_joint : {contact_forces[9]}\nFR_calf_joint : {contact_forces[10]}\nRL_calf_joint : {contact_forces[11]}\nRR_calf_joint : {contact_forces[12]}")
    target_angles= np.deg2rad(controller.read_values())
    go.control_dofs_position(
            target_angles, 
            dofs_idx_local = joints_local_idx[1:]
        )


    scene.step()
    controller.update()




"""
-10 to 10
------
25 to 70
-----
40 to 110
last
-130 to -75
"""