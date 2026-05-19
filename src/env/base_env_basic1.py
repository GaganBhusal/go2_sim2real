import math
import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from genesis.utils.geom import transform_by_quat, inv_quat
import random
from src import DomainRandomization
# from src.utils import Sliders
from tensordict import TensorDict

class WalkENV(gym.Env):

    def __init__(self, render = False, backend = gs.gpu, num_envs = 100, device="cuda", plane = True, t_x = 20, t_y = 50, number_of_lanes=1, number_of_rows=10):
        super().__init__()


        self.device = device
        self.num_envs = num_envs
        self.max_episode_length = 100000


        # Basic Parameters
        self.height_range = 0.3
        self.vx_range = (-0.5, 1.0)
        self.wz_range = (-1.2, 1.2)
        self.pitch_range = (-0.15, 0.15)
        self.action_scale = 0.25
        self.time_step = 0
        self.sigma = 0.25
        self.dt = 0.02
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.reward_survival = 20


        # Terrain Setup
        terrain_x, self.terrain_y = t_x, t_y
        self.start_x, self.start_y, self.start_z = 2, 2, 0.42
        self.num_lanes = number_of_lanes
        self.num_of_rows = number_of_rows
        self.env_seperation = (0, (self.terrain_y * self.num_lanes-5)/self.num_envs)
        self.terrain_length = number_of_rows * t_x
        self.terrain_breadth = self.num_lanes * t_y
        
        # Setting Up Scene and sensors
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=1,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                # max_collision_pairs=20,
            ),
            renderer=gs.renderers.BatchRenderer(use_rasterizer=True),
            show_viewer=render)
        
        if plane:
            plane = self.scene.add_entity(gs.morphs.Plane())
        
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file='//home/yayy/My/Codeeeeee/Simulators/Genesis/genesis-world/genesis/assets/urdf/go2/urdf/go2.urdf'),
            # decimate_fraction = 0.2,
        )

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx = self.robot.idx,
                link_idx_local = self.robot.get_link("base").idx_local,
                draw_debug = True
            )
        )

        # self.lidar = self.scene.add_sensor(
        #     gs.sensors.Lidar(
        #         pattern=gs.sensors.SphericalPattern(
        #         fov=(360.0, 60.0),
        #         # n_points=(128, 32),
        #     ),
        #         draw_debug=False,
        #         entity_idx=self.robot.idx,
        #         link_idx_local = self.robot.get_link("base").idx_local,
        #         pos_offset=(0.25, 0, 0.3),
        #         max_range=100.0,
        #         min_range=0.05,
        #         return_world_frame=False,
        #     )
        # )

        # Adding Camera
        self.cam_forward = self.scene.add_camera(
            res=(64, 64),
            pos=(0, 0.0, 0),
            lookat=(10, 0, 0),
            fov=120,
            GUI=False,
            near = 0.01,
        )
        
        self.offset_T_forward = np.eye(4)
        self.offset_T_forward[0, 3] = 0.31 #x
        self.offset_T_forward[1, 3] = 0.0  #y
        self.offset_T_forward[2, 3] = 0.15 #z

        r = Rotation.from_euler('xyz', [74, 0, -90], degrees=True)
        self.offset_T_forward[:3, :3] = r.as_matrix()
        self.cam_forward.attach(rigid_link=self.robot.get_link("base"), offset_T=self.offset_T_forward)
        # self.cam_sliders_forward = Sliders(values=[0.31, 0, 0.15, 74, 0, -90])
        # self.depth_img = None
        

        # self.cam_foot = self.scene.add_camera(
        #     res=(640, 480),
        #     pos=(0, 0.0, 0),
        #     lookat=(10, 0, 0),
        #     fov=100,
        #     GUI=True,
        # )

        # offset_T_foot = np.eye(4)
        # offset_T_foot[0, 3] = 1
        # offset_T_foot[1, 3] = 0.0
        # offset_T_foot[2, 3] = 0.5
        # r = Rotation.from_euler('z', -90, degrees=True)
        # offset_T_foot[:3, :3] = r.as_matrix()
        # self.cam_foot.attach(rigid_link=self.robot.get_link("base"), offset_T=offset_T_foot)
        # self.cam_sliders_foot = Sliders()



        curriculum_terrains = [
            "flat_terrain",
            "wave_terrain",
            "pyramid_sloped_terrain",
            "pyramid_stairs_terrain",
            "discrete_obstacles_terrain",
            "random_uniform_terrain",
        ]
        
        print(curriculum_terrains)
        print()
        curriculum_terrains = [[random.choice(curriculum_terrains) for _ in range(self.num_lanes)] for i in range(self.num_of_rows)]
        
        if not plane:
            self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    n_subterrains=(self.num_of_rows, self.num_lanes),
                    subterrain_size=(terrain_x, self.terrain_y),      
                    horizontal_scale=0.1,            
                    vertical_scale=0.005,            
                    subterrain_types=curriculum_terrains,
                    randomize=False,
                    name="my_dog_curriculum"
                ),
            )

        self._get_internal_info()
        self._setup_force_sensor()

        self.scene.build(n_envs = self.num_envs, env_spacing = (5.0, 5.0))
        self.robot.set_dofs_kp(torch.tensor([40] * 12),dofs_idx_local = self.joints_local_idx)
        self.robot.set_dofs_kv(torch.tensor([1] * 12),dofs_idx_local = self.joints_local_idx)
        self.domainrandomizer = DomainRandomization(self.robot, self.num_envs, self.joints_local_idx)

        # Observations and actions
        self.num_obs = 47
        self.obs_history_length = 3
        self.total_obs_len = 47
        self.num_actions = 12
        self.cfg = {}

        custom_commands_list = ["Velocity forward", "Angular Velocity XY Plane"]
        self.custom_commands = torch.zeros((self.num_envs, len(custom_commands_list)), device=self.device)
        self.command_envs = torch.zeros(self.num_envs, device=self.device)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_obs_len, ),
            dtype=float
        )
        
        self.obs_histoy = torch.zeros((self.num_envs, self.total_obs_len), device = self.device)
        self.obs_buf = torch.zeros((self.num_envs, self.total_obs_len), device=self.device)

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(12, ),
            dtype=float
        )

        self.__initial_positions = torch.deg2rad(
            torch.tensor([
                0, 0, 0, 0, 50, 50, 55, 55, -90, -90, -90, -90
            ])
        )
        # The hip joints can be set between -90 to -125 for different height based movement and according to requirements.

        self.actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.second_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros((self.num_envs, 12))

        # Actual Joint Limits given in the Unitree Go2 specification.
        self.actual_joint_limits = torch.deg2rad(
            torch.tensor([
                (-48, 48), 
                (-48, 48),
                (-48, 48), 
                (-48, 48), 

                (-90, 200), 
                (-90, 200), 

                (-30, 260), 
                (-30, 260), 

                (-156, -48), 
                (-156, -48), 
                (-156, -48), 
                (-156, -48)
            ])
        )

        # The joint limits defined in the URDF

        self.URDF_specified_joint_limits = torch.deg2rad(
            torch.tensor([
                (-60, 60), 
                (-60, 60),
                (-60, 60), 
                (-60, 60), 

                (-200, 90), 
                (-200, 90), 

                (-260, 30), 
                (-260, 30), 

                (-156, -48), 
                (-156, -48), 
                (-156, -48), 
                (-156, -48)
            ])
        )

        # Joint limits I used while training
        self.joint_limits = torch.deg2rad(
            torch.tensor([
                (-40, 40),
                (-40, 40),
                (-40, 40),
                (-40, 40),

                (-180, 70),
                (-180, 70),

                (-240, 20),
                (-240, 20),

                (-140, -40),
                (-140, -40),
                (-140, -40),
                (-140, -40)
            ])
        )

        self.__get_linear_velocity()
    
        self.extras = dict()        
        self.extras["observations"] = dict()
        self.extras["episode"] = dict()

        self.feet_idx = [
            self.robot.get_link("FL_calf").idx_local,
            self.robot.get_link("FR_calf").idx_local,
            self.robot.get_link("RL_calf").idx_local,
            self.robot.get_link("RR_calf").idx_local,
        ]

        self.trot_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        self.step_freq = 2.0 
        self.target_foot_height = 0.08
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)

        self.episode_sums = {
            "reward_for_tracking_vx": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "reward_for_tracking_wz": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "height_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "pitch_penalty" : torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "lin_vel_z_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "roll_pitch_velocity_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "action_rate_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "second_order_action_rate": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "similar_to_default": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "torque_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "joint_vel_penalty" : torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "out_of_range_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "collision_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # "feet_air_time": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # "foot_slip_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # "swing_phase_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # "stance_phase_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # "foot_swing_height_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # "raibert_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "reward_survival" : torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "Reward_Per_Environment": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        
        self.update_cam()

    def _setup_force_sensor(self):
        self.contact_sensors = []
        for idx in range(13):
            contact_sensor = self.scene.add_sensor(
                gs.sensors.ContactForce(
                    entity_idx=self.robot.idx,
                    link_idx_local=idx,
                    min_force = 1.0,
                    draw_debug = False,
                )
            )
            self.contact_sensors.append(contact_sensor)


    def __get_linear_velocity(self):
        vel_body = self.robot.get_link("base").get_vel()
        base_quat = self.robot.get_link("base").get_quat()
        vel_body = transform_by_quat(vel_body, inv_quat(base_quat))
        return torch.tensor(vel_body)


    def _get_internal_info(self):
        self.joints_local_idx = []
        for links in self.robot.links[1:]:
            joints = links.joints
            for joint in joints:
                print(f"Joint name: {joint.name}, local index: {joint.dof_idx_local}")
                self.joints_local_idx.append(joint.dof_idx_local)
                


    def _get_imu_values(self):
        imu_value = self.imu.read()
        linear_acceleration, _angular_vel, magnetometer = self.imu.read()

        return linear_acceleration, _angular_vel

    def _get_ypr(self):
        quat_wxyz = self.robot.get_link("base").get_quat()
        quat_xyzw = torch.stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]], dim = 1)
        rotation = Rotation.from_quat(quat_xyzw.detach().cpu().numpy())
        euler_angles = rotation.as_euler("xyz")
        return torch.tensor(np.array(euler_angles), dtype=torch.float)

    def _get_collision_forces(self):
        collision_forces = []
        for sensor in self.contact_sensors[:-4]:
            contact_forces = sensor.read()
            collision_forces.append(contact_forces.norm(dim=-1))
        return torch.stack(collision_forces, dim=1)

    def _calculate_projected_acceleration(self):
        quat = self.robot.get_link("base").get_quat()
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        gx = 2 * (x*z - w*y)
        gy = 2 * (y*z + w*x)
        gz = 1 - 2 * (x*x + y*y)
        return torch.stack([-gx, -gy, -gz], dim=1)

    def _calculate_reward(self):
        # Values for calculating Reward
        base_pos = self.robot.get_link("base").get_pos()
        base_vel = self.robot.get_link("base").get_vel()
        _, base_ang_vel = self._get_imu_values()
        dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx)
        dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)
        torques = self.robot.get_dofs_control_force(dofs_idx_local=self.joints_local_idx)

        base_quat = self.robot.get_link("base").get_quat()
        base_lin_vel_body = self.__get_linear_velocity()

        vx = base_lin_vel_body[:, 0]
        vy = base_lin_vel_body[:, 1]
        wz = base_ang_vel[:, 2]
        vz = base_lin_vel_body[:, 2]
        base_x_pos = base_pos[:, 0]
        base_y_pos = base_pos[:, 1]
        base_height = base_pos[:, 2]
        current_pitch = self._get_ypr()[:, 1]

        target_air_time = 0.4
        contact_forces = self.robot.get_links_net_contact_force()
        foot_forces_z = contact_forces[:, self.feet_idx, 2]
        # print(foot_forces_z[0])
        contact = foot_forces_z > 1.0
        first_contact = (self.feet_air_time > 0.0) & contact

        fl_vel = self.robot.get_link("FL_calf").get_vel()
        fr_vel = self.robot.get_link("FR_calf").get_vel()
        rl_vel = self.robot.get_link("RL_calf").get_vel()
        rr_vel = self.robot.get_link("RR_calf").get_vel()
        foot_velocities = torch.stack([fl_vel, fr_vel, rl_vel, rr_vel], dim=1)
        foot_position_z = torch.stack([
            self.robot.get_link("FL_calf").get_pos()[:, 2],
            self.robot.get_link("FR_calf").get_pos()[:, 2],
            self.robot.get_link("RL_calf").get_pos()[:, 2],
            self.robot.get_link("RR_calf").get_pos()[:, 2],
        ], dim=1)

        current_time = self.episode_length_buf * self.dt
        leg_phases = (current_time.unsqueeze(1) * self.step_freq + self.trot_offsets) % 1.0
        swing_progress = (leg_phases - 0.5) * 2.0 * torch.pi
        target_z = torch.clamp(torch.sin(swing_progress) * self.target_foot_height, min=0.0)
        swing_phase = leg_phases >= 0.5
        stance_phase = leg_phases < 0.5
        foot_xy_vel_penalty = torch.square(foot_velocities[:, :, 0] - self.custom_commands[:, 0].unsqueeze(1) * 2.0) + torch.square(foot_velocities[:, :, 1] - torch.zeros_like(self.custom_commands[:, 0]).unsqueeze(1))
        # print(swing_phase[0])

        collision_forces = self._get_collision_forces()
                
        # Rewards and Penalty for Custom Commands
        reward_for_tracking_vx = torch.exp(-torch.square(self.custom_commands[:, 0] - vx) / self.sigma)
        reward_for_tracking_wz = torch.exp(-torch.square(self.custom_commands[:, 1] - wz) / self.sigma)
        height_penalty = torch.square(base_height - self.height_range)
        # pitch_penalty = torch.square(current_pitch - self.custom_commands[:, 2])
        
        # Additional Penaties
        lin_vel_z_penalty = torch.square(vz)
        roll_pitch_velocity_penalty = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        action_rate_penalty = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        second_order_action_rate = torch.sum(torch.square(self.second_last_actions - 2 * self.last_actions + self.actions), dim=1)
        reward_similar_to_default = torch.sum(torch.square(dof_pos - self.__initial_positions), dim=1)
        torque_penalty = torch.sum(torch.square(torques), dim=1)
        joint_vel_penalty = torch.sum(torch.square(dof_vel), dim=1)
        out_of_range_penalty = torch.sum((dof_pos < self.joint_limits[:, 0]) | (dof_pos > self.joint_limits[:, 1]), dim=1)
        collision_penalty = (collision_forces > 1.0).any(dim=-1)
        
        # reward_feet_air_time = torch.sum(torch.exp(-torch.square(self.feet_air_time - target_air_time) / (0.1)) * first_contact, dim=1)
        # foot_slip_penalty = torch.sum(torch.square(foot_velocities[:, :, :2]) * contact.unsqueeze(-1), dim=[1, 2])

        # swing_phase_penalty = torch.sum(torch.square(foot_forces_z * swing_phase.float()), dim=1)
        # stance_phase_penalty = torch.sum(torch.square(foot_velocities[:, :, :2] * stance_phase.unsqueeze(-1).float()), dim=[1, 2])
        # foot_swing_height_penalty = torch.sum(torch.square(target_z - (foot_position_z - 0.21)), dim=1)
        # raibert_penalty = torch.sum(foot_xy_vel_penalty * swing_phase.float(), dim=1)

        # print(
        #     reward_for_tracking_vx.shape,
        #     reward_for_tracking_wz.shape,
        #     height_penalty.shape,
        #     # pitch_penalty.shape,
        #     lin_vel_z_penalty.shape,
        #     roll_pitch_velocity_penalty.shape,
        #     action_rate_penalty.shape,
        #     second_order_action_rate.shape,
        #     reward_similar_to_default.shape,
        #     torque_penalty.shape,
        #     joint_vel_penalty.shape,
        #     out_of_range_penalty.shape,
        #     collision_penalty.shape,
            # reward_feet_air_time.shape,
            # foot_slip_penalty.shape,
            # swing_phase_penalty.shape,
            # stance_phase_penalty.shape,
            # foot_swing_height_penalty.shape,
            # raibert_penalty.shape
        # )

        self.feet_air_time += self.dt       
        self.feet_air_time[contact] = 0.0

        reward_for_tracking_vx = + 3.0 * reward_for_tracking_vx
        reward_for_tracking_wz = + 2 * reward_for_tracking_wz
        height_penalty = - 30.0 * height_penalty
        # pitch_penalty = - 5.0 * pitch_penalty
        lin_vel_z_penalty = - 0.02 * lin_vel_z_penalty
        roll_pitch_velocity_penalty = - 0.001 * roll_pitch_velocity_penalty
        action_rate_penalty = - 0.01 * action_rate_penalty
        second_order_action_rate = - 0.01 * second_order_action_rate
        reward_similar_to_default = - 0.01 * reward_similar_to_default
        torque_penalty = - 0.001 * torque_penalty
        joint_vel_penalty = - 0.005 * joint_vel_penalty
        out_of_range_penalty = - 0.5 * out_of_range_penalty
        collision_penalty = - 1.0 * collision_penalty

        # reward_feet_air_time = + 1.0 * reward_feet_air_time
        # foot_slip_penalty = - 5 * foot_slip_penalty
        # swing_phase_penalty = - 0.05 * swing_phase_penalty
        # stance_phase_penalty = -0.5 * stance_phase_penalty
        # foot_swing_height_penalty = - 5.0 * foot_swing_height_penalty
        # raibert_penalty = -0.5 * raibert_penalty
        reward_survival = torch.ones_like(base_height) * 1.0

        reward = (
            reward_for_tracking_vx
            + reward_for_tracking_wz
            + height_penalty
            # + pitch_penalty
            + lin_vel_z_penalty
            + roll_pitch_velocity_penalty
            + action_rate_penalty
            + second_order_action_rate
            + reward_similar_to_default
            + torque_penalty
            + joint_vel_penalty
            + out_of_range_penalty
            + collision_penalty
            # + reward_feet_air_time
            # + foot_slip_penalty
            # + swing_phase_penalty
            # + stance_phase_penalty
            # + foot_swing_height_penalty
            # + raibert_penalty
            + reward_survival
        )


        euler = self._quat_to_euler(base_quat)
        terminated = ((torch.abs(euler[:, 0]) > 0.5) | (torch.abs(euler[:, 1]) > 0.5) | (base_height < 0.18))
        out_of_trajectory = (base_x_pos < 0.5) | (base_x_pos > self.terrain_length-0.5) | (base_y_pos < 0.5) | (base_y_pos > self.terrain_breadth - 0.5)
        out_of_trajectory = False
        reward[terminated] -= 500

        
        self.episode_sums["reward_for_tracking_vx"] += reward_for_tracking_vx
        self.episode_sums["reward_for_tracking_wz"] += reward_for_tracking_wz
        self.episode_sums["height_penalty"] += height_penalty
        # self.episode_sums["pitch_penalty"] += pitch_penalty
        self.episode_sums["lin_vel_z_penalty"] += lin_vel_z_penalty
        self.episode_sums["roll_pitch_velocity_penalty"] += roll_pitch_velocity_penalty
        self.episode_sums["action_rate_penalty"] += action_rate_penalty
        self.episode_sums["second_order_action_rate"] += second_order_action_rate
        self.episode_sums["similar_to_default"] += reward_similar_to_default
        self.episode_sums["torque_penalty"] += torque_penalty
        self.episode_sums["joint_vel_penalty"] += joint_vel_penalty
        self.episode_sums["out_of_range_penalty"] += out_of_range_penalty
        self.episode_sums["collision_penalty"] += collision_penalty
        # self.episode_sums["feet_air_time"] += reward_feet_air_time
        # self.episode_sums["foot_slip_penalty"] += foot_slip_penalty
        # self.episode_sums["swing_phase_penalty"] += swing_phase_penalty
        # self.episode_sums["stance_phase_penalty"] += stance_phase_penalty
        # self.episode_sums["foot_swing_height_penalty"] += foot_swing_height_penalty
        # self.episode_sums["raibert_penalty"] += raibert_penalty
        self.episode_sums["reward_survival"] += reward_survival
        self.episode_sums["Reward_Per_Environment"] += reward * 1.0
        reward *= self.dt

        return reward, terminated, out_of_trajectory

    def get_observations(self):
        obs_buf = self._get_obs()
        return TensorDict(
        {
            "policy": obs_buf["policy"],
            "image": obs_buf["image"],
        },
        batch_size=self.num_envs)

    def get_privileged_observations(self):
        return None

    def _get_obs(self):
        
        self.base_pos = self.robot.get_link("base").get_pos()
        self.linear_acc, self.base_ang_vel = self._get_imu_values()
        self.base_quat = self.robot.get_link("base").get_quat()
        self.base_lin_vel_body = self.__get_linear_velocity()
        projected_gravity = self._calculate_projected_acceleration()
        dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx) - self.__initial_positions
        dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)

        current_time = self.episode_length_buf * self.dt
        leg_phases = (current_time.unsqueeze(1) * self.step_freq + self.trot_offsets) % 1.0
        phase_angles = leg_phases * 2 * torch.pi
        clock_obs = torch.cat([torch.sin(phase_angles), torch.cos(phase_angles)], dim=1)
        
        scaled_commands = torch.stack([
            self.custom_commands[:, 0] * 1.0,   
            self.custom_commands[:, 1] * 0.5,
        ], dim=1)

        # _, depth_img, _, _ = self.cam_forward.render(depth=True)
        # print(torch.tensor(depth_img, dtype=torch.float32).unsqueeze(1).shape)

        current_obs = TensorDict({          
            "policy": torch.cat([
            self.base_lin_vel_body,                        # 3 
            self.base_ang_vel * 0.5,                       # 3
            projected_gravity ,                       # 3
            dof_pos,                                  # 12
            dof_vel * 0.05,                           # 12
            self.actions,                             # 12
            scaled_commands,                          # 3
            ], dim=1),
            "image": torch.tensor(self.depth_img, dtype=torch.float32).unsqueeze(1) # n_envs x 1 x 64 x 64
        }, batch_size=self.num_envs, device=self.device)

        # print(current_obs["image"].shape)

        # Add history of observations
        # self.obs_histoy[:, :-self.num_obs] = self.obs_histoy[:, self.num_obs:].clone()
        # self.obs_histoy[:, -self.num_obs:] = current_obs

        return current_obs

    def _reset_idx(self, envs_idx):
        
        if len(envs_idx) == 0:
            return
        
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            avg_reward = torch.mean(value[envs_idx]) / self.max_episode_length
            self.extras["episode"]["rew_" + key] = avg_reward
            
            value[envs_idx] = 0.0

        self.robot.set_dofs_position(
            self.__initial_positions.repeat(len(envs_idx), 1),
            dofs_idx_local=self.joints_local_idx,
            envs_idx=envs_idx
        )
        self.robot.set_dofs_velocity(
            torch.zeros((len(envs_idx), 12), device=self.device),
            dofs_idx_local=self.joints_local_idx,
            envs_idx=envs_idx
        )

        self.episode_length_buf[envs_idx] = 0
        self.actions[envs_idx] = 0
        self.last_actions[envs_idx] = 0
        self.second_last_actions[envs_idx] = 0
        self.last_dof_vel[envs_idx] = 0
        self.obs_histoy[envs_idx] = 0

        x = torch.randint(low=10, high=self.terrain_length-10, size=(len(envs_idx), ))
        # y = torch.randint(low=20, high=self.terrain_breadth-20, size=(len(envs_idx), ))
        z = torch.ones_like(x) * 0.4
        x = 0
        y = 0
        z = 0.4
        
        # a = torch.ones_like(x)
        # b = torch.zeros_like(a)
        # c = torch.zeros_like(a)
        # d = torch.zeros_like(a)

        base_quat = torch.tensor([1, 0, 0, 0], device=self.device)
        self.robot.set_quat(quat=base_quat, envs_idx=envs_idx)
        base_pos = torch.tensor([x, y, z], device=self.device)
        # base_pos = torch.stack([x, y, z], dim=1)
        self.robot.set_pos(base_pos, envs_idx=envs_idx)
        self.domainrandomizer.randomize(envs_idx)

        self.update_commands(envs_idx)
   
    def update_cam(self):
        
        # offset_T_forward = self.cam_sliders_forward.update_values()

        base_link = self.robot.get_link("base")
        self.cam_forward.attach(base_link, self.offset_T_forward)
        self.cam_forward.move_to_attach()

        _, self.depth_img, _, _ = self.cam_forward.render(rgb=False, depth=True, segmentation=False, colorize_seg=False, normal=False) 


    def reset(self):
        all_idx = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_idx)
        self.scene.reset()
        self.obs_buf = self._get_obs()
        return self.obs_buf

    def update_commands(self, idx):
    
        self.custom_commands[idx, 0] = torch.rand(len(idx), device=self.device) * (self.vx_range[1] - self.vx_range[0]) + self.vx_range[0]
        self.custom_commands[idx, 1] = torch.rand(len(idx), device=self.device) * (self.wz_range[1] - self.wz_range[0]) + self.wz_range[0]

        self.command_envs[idx] = 0

    def step(self, action):

        self.time_step += 1
        self.actions = action.clone()

        target_dof_pos = self.__initial_positions + action * self.action_scale
        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local = self.joints_local_idx)
        self.scene.step()
        self.command_envs += 1
    
        reward, terminated, out_of_trajectory = self._calculate_reward()

        self.episode_length_buf += 1
        time_out = self.episode_length_buf >= self.max_episode_length
        self.extras["time_outs"] = time_out
        dones = terminated | time_out | out_of_trajectory

        final_obs = self._get_obs().clone()
        
        self.second_last_actions = self.last_actions.clone()
        self.last_actions = self.actions.clone()

        self.previous_joint_velocity = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx)

        reset_ids = torch.nonzero(dones).flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        observation = self._get_obs()
        
        # lidar_data = self.lidar.read()

        info = {
            "final_obs":   final_obs.detach().cpu().numpy(),
            "terminated":  terminated.detach().cpu().numpy(),
            # "lidar_points": lidar_data[0][0],          # (N, 3) tensor, env 0
            "robot_pos":    self.robot.get_link("base").get_pos()[0],
            "robot_quat":   self.robot.get_link("base").get_quat()[0],
        }

        self.update_cam()
        self.extras["camera"] = self.depth_img

        return observation, reward, dones, self.extras   

    def _quat_to_euler(self, q):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(torch.abs(sinp) >= 1,
                            torch.sign(sinp) * torch.pi / 2,
                            torch.asin(sinp))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=-1)
