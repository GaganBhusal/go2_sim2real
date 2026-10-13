import math
import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from genesis.utils.geom import transform_by_quat, inv_quat
import random
from src.utils import DomainRandomization
from tensordict import TensorDict
from src.utils.math_utils import _quat_to_euler, _calculate_projected_acceleration, _euler_to_quat

class BaseQuadrupedEnv(gym.Env):

    def __init__(self, config, render=False, backend=None, num_envs=100, device="cuda", plane=True, t_x=20, t_y=50, number_of_lanes=20, number_of_rows=50, domain_randomize=False, terrain=False, test=False):
        super().__init__()
        
        self.test = test
        self.config = config
        self.plane = plane
        self.num_envs = self.config.num_envs
        self.device = self.config.device
        self.max_episode_length = self.config.max_episode_length
        self.episode_length_buf = torch.zeros(self.config.num_envs, dtype=torch.int, device=self.config.device)
        self.horizontal_scale = 0.05
        self.vertical_scale = 0.005
        self.start_x, self.start_y, self.start_z = 2, 2, 0.42

        self.terrain = terrain
        if terrain:
            self.t_x = t_x
            self.t_y = t_y
            self.number_of_lanes = number_of_lanes
            self.number_of_rows = number_of_rows
            self.env_seperation = (0, (self.t_x * self.t_y-5)/self.config.num_envs)
            self.terrain_length = self.number_of_rows * t_x
            self.terrain_breadth = self.number_of_lanes * t_y

            self.respawn_x_range = (self.start_x, self.terrain_length - self.start_x)
            self.respawn_y_range = (self.start_y, self.terrain_breadth - self.start_y)
            self.respawn_z_range = (0.25, 0.42)

        self._setup_scene(render)
        self._get_internal_info()
        self._setup_obs_and_action_spaces()

        self.feet_indices = self.config.FOOTS
        self.feet_air_time = torch.zeros(self.config.num_envs, len(self.config.FOOTS), device=self.config.device)
        self.feet_contact_time = torch.zeros(self.num_envs, 4, device=self.device)   # NEW
        self.last_contacts = torch.zeros(self.config.num_envs, len(self.config.FOOTS), dtype=torch.bool, device=self.config.device)
        self.peak_heights = torch.zeros(self.config.num_envs, len(self.config.FOOTS), device = self.config.device)

        self.randomize = domain_randomize
        if self.randomize:
            self.domainrandomizer = DomainRandomization(robot=self.robot, joints_local_idx=self.joints_local_idx_flat, config=self.config)
            print(f"Domain Randomization Enabled!!")


        self.episode_sums = self._init_episode_sums()

    def _init_episode_sums(self):
        return {
            "Reward_Per_Environment": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
        }
        
    def _fetch_configs(self):
        pass
    

    def _setup_scene(self, render):
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(
                dt=self.config.dt,
                substeps=self.config.sim_substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=self.config.enable_self_collision,
                tolerance=1e-5,
                max_collision_pairs=self.config.max_collision_pairs,
                batch_dofs_info=True,
                batch_links_info=True,
            ),
            show_viewer=render)
        
        if self.terrain:
            curriculum_terrains = [
                "flat_terrain",
                "fractal_terrain",
                "wave_terrain",
                "pyramid_sloped_terrain",
                "pyramid_stairs_terrain",
                "discrete_obstacles_terrain",
                "random_uniform_terrain",
            ]

            curriculum_terrains = [[random.choice(curriculum_terrains) for _ in range(self.number_of_lanes)] for i in range(self.number_of_rows)]
            self.random_terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    n_subterrains=(self.number_of_rows, self.number_of_lanes),
                    subterrain_size=(self.t_y, self.t_y),      
                    horizontal_scale=self.horizontal_scale,            
                    vertical_scale=self.vertical_scale,            
                    subterrain_types=curriculum_terrains,
                    randomize=False,
                    name="my_dog_curriculum"
                ),
            )
        else:
            plane = self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=self.config.URDF_LOCATION),
        )

        # Build list of dof_idx_local for internal info logic
        self.joints_local_idx = []
        for links in self.robot.links[1:]:
            print(f"Link Name: {links.name} \t Link Local Idx: {links.idx_local}")
            joints = links.joints
            for joint in joints:
                print(f"Joint Name: {joint.name} \t Joint DoF Local Idx: {joint.dofs_idx_local}")
                if hasattr(joint, 'dofs_idx_local'):
                    self.joints_local_idx.append(joint.dofs_idx_local)
                else:
                    self.joints_local_idx.append(joint.dof_idx_local)

        flat_idx = []
        for item in self.joints_local_idx:
            if isinstance(item, (list, tuple)):
                flat_idx.extend(item)
            else:
                flat_idx.append(item)
        self.joints_local_idx_flat = flat_idx

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx = self.robot.idx,
                link_idx_local = self.robot.get_link(self.config.BASE).idx_local,
                draw_debug = False
            )
        )

# NEED TO CHANGE HERE
        self.contact_bool_sensors = []
        self.contact_force_sensors = []
        
        # for idx in range(len(self.joints_local_idx_flat)):
        #     contact_sensor = self.scene.add_sensor(
        #         gs.sensors.Contact(
        #             entity_idx = self.robot.idx,
        #             link_idx_local = idx,
        #             draw_debug = False
        #         )
        #     )
        #     self.contact_bool_sensors.append(contact_sensor)


        for foot_name in self.config.FOOT_LINK_NAMES:
            print(foot_name)
            foot_link = self.robot.get_link(foot_name)
        # for idx in self.config.FOOTS:
            contact_force = self.scene.add_sensor(
                gs.sensors.ContactForce(
                    entity_idx=self.robot.idx,
                    link_idx_local=foot_link.idx_local,
                    min_force=1.0,
                    draw_debug=False,
                )
            )
            self.contact_force_sensors.append(contact_force)

        self.scene.build(n_envs = self.config.num_envs, env_spacing = (5.0, 5.0))

# TILL HERE


    def _setup_obs_and_action_spaces(self):
        custom_commands_list = ["Velocity forward", "Vy Velocity", "Angular Velocity XY Plane"]
        self.custom_commands = torch.zeros((self.config.num_envs, len(custom_commands_list)), device=self.config.device)
        self.command_envs = torch.zeros(self.config.num_envs, device=self.config.device)

        self.num_actions = self.config.total_actions
        self.cfg = {}
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(self.total_obs_len, ),
        #     dtype=float
        # )
        
        # self.action_space = gym.spaces.Box(
        #     low=-1,
        #     high=1,
        #     shape=(self.num_actions, ),
        #     dtype=float
        # )

        self.actions = torch.zeros((self.config.num_envs, self.num_actions), device=self.config.device)
        self.last_actions = torch.zeros_like(self.actions, device=self.config.device)
        self.second_last_actions = torch.zeros_like(self.actions, device=self.config.device)
        self.last_dof_vel = torch.zeros((self.config.num_envs, self.num_actions), device=self.config.device)
        
        self.sine_phase = torch.zeros((self.config.num_envs, 1), device=self.config.device)
        self.cosine_phase = torch.zeros((self.config.num_envs, 1), device=self.config.device)

        self.extras = dict()        
        self.extras["observations"] = dict()
        self.extras["episode"] = dict()

    def __get_linear_velocity(self):
        vel_body = transform_by_quat(self.vel_body, inv_quat(self.base_quat))
        return vel_body


# LOOK INTO THIS TOO
    def _get_internal_info(self):
        kp_gains = torch.zeros(self.config.total_actions, device=self.config.device)
        kv_gains = torch.zeros(self.config.total_actions, device=self.config.device)
        
        default_kp = self.config.kp
        default_kv = self.config.kv


        for idx, links in enumerate(self.robot.links[1:]):
            for joint in links.joints:
                kp_gains[idx] = default_kp
                kv_gains[idx] = default_kv

        self.robot.set_dofs_kp(kp_gains, dofs_idx_local=self.joints_local_idx_flat)
        self.robot.set_dofs_kv(kv_gains, dofs_idx_local=self.joints_local_idx_flat)

    def _get_imu_values(self):
        linear_acceleration, _angular_vel, magnetometer = self.imu.read()
        return linear_acceleration, _angular_vel


# HERE THESE TWO TOO
    # def _get_contacts_bool(self):
    #     contacts = []
    #     for sensor in self.contact_bool_sensors:
    #         contacts.append(sensor.read())
    #     return torch.stack(contacts, dim=1) if len(contacts) > 0 else torch.zeros((self.config.num_envs, 0, 1), device=self.device)

    def _get_collision_forces(self):
        collision_forces = []
        for sensor in self.contact_force_sensors:
            contact_forces = sensor.read()
            collision_forces.append(contact_forces)
        return torch.stack(collision_forces, dim=1) if len(collision_forces) > 0 else torch.zeros((self.config.num_envs, 0, 3), device=self.device)

    def get_local_height_of_robot(self):
        if not self.terrain:
            return self.base_z
            
        hf_np = self.random_terrain.geoms[0].metadata["height_field"] 
        self.hf = torch.tensor(hf_np, dtype=torch.float32, device=self.device)
        
        hf = self.hf 
        hs = self.horizontal_scale
        vs = self.vertical_scale

        x_idx = self.base_x / hs
        y_idx = self.base_y / hs

        x0 = x_idx.long()
        y0 = y_idx.long()
        x1 = x0 + 1
        y1 = y0 + 1

        H, W = hf.shape
        x0 = x0.clamp(0, H - 1)
        y0 = y0.clamp(0, W - 1)
        x1 = x1.clamp(0, H - 1)
        y1 = y1.clamp(0, W - 1)

        tx = (x_idx - x0.float()).clamp(0.0, 1.0)
        ty = (y_idx - y0.float()).clamp(0.0, 1.0)

        h00 = hf[x0, y0] * vs
        h10 = hf[x1, y0] * vs
        h01 = hf[x0, y1] * vs
        h11 = hf[x1, y1] * vs

        upper_left = tx <= ty

        h_upper = h00 * (1.0 - ty) + h01 * (ty - tx) + h11 * tx
        h_lower = h00 * (1.0 - tx) + h10 * (tx - ty) + h11 * ty

        height = torch.where(upper_left, h_upper, h_lower)

        return self.base_z - height

    def _get_all_values_at_once(self):
        self.base_pos = self.robot.get_link(self.config.BASE).get_pos()
        _, self.base_ang_vel = self._get_imu_values() 
        self.base_quat = self.robot.get_link(self.config.BASE).get_quat() 
        self.euler = _quat_to_euler(self.base_quat) 
        self.vel_body = self.robot.get_link(self.config.BASE).get_vel() 
        self.base_lin_vel_body = self.__get_linear_velocity()
        self.projected_gravity = _calculate_projected_acceleration(self.base_quat)
        self.dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx_flat) 
        self.dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx_flat) 
        self.torques = self.robot.get_dofs_control_force(dofs_idx_local=self.joints_local_idx_flat)
        self.collision_forces = self._get_collision_forces()
        

        calf_pos = torch.stack(
            [self.robot.get_link(name).get_pos() for name in self.config.FOOT_LINK_NAMES],
            dim=1
        )
        calf_quat = torch.stack(
            [self.robot.get_link(name).get_quat() for name in self.config.FOOT_LINK_NAMES],
            dim=1
        )
        local_foot_offset = torch.tensor([0.0, 0.0, -0.235], device=self.device)
        self.feet_pos = calf_pos + transform_by_quat(
            local_foot_offset.unsqueeze(0).unsqueeze(0).expand_as(calf_pos),
            calf_quat
        )

        self.feet_vel = torch.stack(
                [self.robot.get_link(name).get_vel() for name in self.config.FOOT_LINK_NAMES],
                dim=1
            )
        
        self.base_x = self.base_pos[:, 0]
        self.base_y = self.base_pos[:, 1]
        self.base_z = self.base_pos[:, 2]
        if self.terrain:
            self.base_z = self.get_local_height_of_robot()
            self.base_height = self.base_z
        else:
            self.base_height = self.base_pos[:, 2]
        self.vx = self.base_lin_vel_body[:, 0]
        self.vy = self.base_lin_vel_body[:, 1]
        self.wz = self.base_ang_vel[:, 2]
        self.vz = self.base_lin_vel_body[:, 2]
        self.base_height = self.base_pos[:, 2]
        self.current_pitch = self.euler[:, 1]
        self.current_yaw = self.euler[:, 2]

        # self.whole_body_contacts = abs(self.collision_forces[:, :, 2]) > 1.0
        # # print(self.whole_body_contacts.shape)
        # # self.feet_contacts = abs(self.collision_forces[:, self.config.FOOTS, 2]) > 1.0
        # self.feet_contacts = self.whole_body_contacts[:, self.config.FOOTS - 1]
        # # print(self.feet_contacts.shape)

        # self.whole_body_contacts = abs(self.collision_forces[:, :, 2]) > 1.0
        # print(self.whole_body_contacts.shape)
        # self.feet_contacts = abs(self.collision_forces[:, self.config.FOOTS, 2]) > 1.0
        # self.feet_contacts = self.whole_body_contacts[:, self.config.FOOTS - 1]

        self.feet_contacts = abs(self.collision_forces[:, :, 2]) > 1.0

        period = self.config.gait_period
        offsets = torch.tensor(self.config.gait_offset, device=self.device)  # [0, 0.5, 0.5, 0] -> shape (4,)

        self.phase = (self.episode_length_buf * self.config.dt) % period / period  # shape (num_envs,)

        self.leg_phase = (self.phase.unsqueeze(1) + offsets.unsqueeze(0)) % 1

        self.sine_phase = torch.sin(2 * torch.pi * self.phase)    # (num_envs, 4)
        self.cosine_phase = torch.cos(2 * torch.pi * self.phase) 

    def _calculate_reward(self):
        pass

    def _get_obs(self):
        pass

    def get_observations(self):
        self._get_all_values_at_once()
        self.obs_buf = self._get_obs()
        return self.obs_buf

    def _reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            avg_reward = torch.mean(value[envs_idx]) / self.config.max_episode_length
            self.extras["episode"]["rew_" + key] = avg_reward
            value[envs_idx] = 0.0

        rand_dof = self.config.initial_positions.repeat(len(envs_idx), 1)
        if self.randomize:
            rand_dof += torch.randn_like(rand_dof) * 0.1
        
        rand_dof = torch.clamp(rand_dof,
            self.config.clamped_joint_limits[:, 0],
            self.config.clamped_joint_limits[:, 1])
        
        self.robot.set_dofs_position(
            rand_dof,
            dofs_idx_local=self.joints_local_idx_flat,
            envs_idx=envs_idx
        )

        self.robot.set_dofs_velocity(
            torch.randn((len(envs_idx), self.config.total_actions), device=self.config.device) * 0.1,
            dofs_idx_local=self.joints_local_idx_flat,
            envs_idx=envs_idx
        )

        respawn_pos = torch.zeros((len(envs_idx), 3), device=self.device)
        respawn_pos[:, 2] = 0.35
        if self.randomize:
            respawn_pos[:, 2] = 0.14 * torch.rand(len(envs_idx), device=self.device) + 0.28
        euler = torch.zeros((len(envs_idx), 3), device=self.device)
        


        euler[:, 0] = self.config.roll_range[0] + torch.rand(len(envs_idx), device=self.device) * (self.config.roll_range[1] - self.config.roll_range[0])
        euler[:, 1] = self.config.pitch_range[0] + torch.rand(len(envs_idx), device=self.device) * (self.config.pitch_range[1] - self.config.pitch_range[0])
        euler[:, 2] = self.config.yaw_range[0] + torch.rand(len(envs_idx), device=self.device) * (self.config.yaw_range[1] - self.config.yaw_range[0])

        respawn_quat = _euler_to_quat(euler)

        self.episode_length_buf[envs_idx] = 0
        self.actions[envs_idx] = 0
        self.last_actions[envs_idx] = 0
        self.second_last_actions[envs_idx] = 0
        self.last_dof_vel[envs_idx] = 0
        
        self.feet_air_time[envs_idx] = 0.0
        self.last_contacts[envs_idx] = False
        self.feet_contact_time[envs_idx] = 0.0
        self.peak_heights[envs_idx] = 0.0

        self.sine_phase[envs_idx] = 0
        self.cosine_phase[envs_idx] = 0

        self.robot.set_quat(quat=respawn_quat, envs_idx=envs_idx)
        self.robot.set_pos(respawn_pos, envs_idx=envs_idx)

        if self.randomize:
            self.domainrandomizer.randomize(envs_idx)
            self.domainrandomizer.action_buffer[envs_idx] = 0

        self.update_commands(envs_idx)

    def reset(self):
        all_idx = torch.arange(self.config.num_envs, device=self.config.device)
        self._reset_idx(all_idx)
        self.scene.reset()
        self._get_all_values_at_once()
        self.obs_buf = self._get_obs()
        return self.obs_buf

    def update_commands(self, idx):
        self.custom_commands[idx, 0] = torch.rand(len(idx), device=self.config.device) * (self.config.vx_range[1] - self.config.vx_range[0]) + self.config.vx_range[0]
        self.custom_commands[idx, 1] = torch.rand(len(idx), device=self.config.device) * (self.config.vy_range[1] - self.config.vy_range[0]) + self.config.vy_range[0]
        self.custom_commands[idx, 2] = torch.rand(len(idx), device=self.config.device) * (self.config.wz_range[1] - self.config.wz_range[0]) + self.config.wz_range[0]
        self.custom_commands[idx[torch.rand(len(idx), device=self.config.device) < 0.25]] = 0.0
        
    def step(self, action):
        self.actions = action.clone()
        if self.randomize:
            action = self.domainrandomizer._add_action_latency(self.actions)

        # action = 0.7 * action + 0.3 * self.last_actions

        target_dof_pos = self.config.initial_positions + action * self.config.action_scale
        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local = self.joints_local_idx_flat)
        self.scene.step()

        self._get_all_values_at_once()

        if self.randomize:
            self.domainrandomizer._random_force_push()

        reward, terminated, out_of_trajectory = self._calculate_reward()

        self.episode_length_buf += 1
        time_out = self.episode_length_buf >= self.config.max_episode_length
        
        self.extras["time_outs"] = time_out
        dones = terminated | time_out | out_of_trajectory

        final_obs = self._get_obs().clone()
        
        self.second_last_actions = self.last_actions.clone()
        self.last_actions = self.actions.clone()

        self.previous_joint_velocity = self.dof_vel.clone()
        self.last_dof_vel = self.dof_vel.clone()


        reset_ids = torch.nonzero(dones).flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)
            
        self.obs_buf = self._get_obs()
        observation = self.obs_buf
        
        info = {
            "final_obs":   final_obs["policy"],
            "terminated":  terminated,
            "robot_pos":    self.base_pos[0],
            "robot_quat":   self.base_quat[0],
        }

        return observation, reward, dones, self.extras
