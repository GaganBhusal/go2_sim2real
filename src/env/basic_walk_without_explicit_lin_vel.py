import math
import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from genesis.utils.geom import transform_by_quat, inv_quat
import random
from src.utils import DomainRandomization
# from src.utils import Sliders
from tensordict import TensorDict
from src.utils.math_utils import _quat_to_euler, _calculate_projected_acceleration

class WalkWithoutVx(gym.Env):

    def __init__(self, config, render = False, backend = gs.gpu, num_envs = 100, device="cuda", plane = True, t_x = 20, t_y = 50, number_of_lanes=20, number_of_rows=50, domain_randomize = False, terrain = True):
        super().__init__()

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

        self.randomize = domain_randomize
        if self.randomize:
            self.domainrandomizer = DomainRandomization(self.robot, self.config.num_envs, self.joints_local_idx, device=self.config.device)

        # self.robot.set_friction_ratio(
        #     friction_ratio=torch.tensor([0.1] * self.robot.n_links, device=self.config.device),
        #     links_idx_local=range(self.robot.n_links),
        # )

        self.episode_sums = {
            "reward_for_tracking_vx": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_for_tracking_vy": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_for_tracking_wz": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),

            # "vy_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "height_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "pitch_penalty" : torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "yaw_penalty" : torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "hip_deviation_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "lin_vel_z_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "roll_pitch_velocity_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "action_rate_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "second_order_action_rate": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "similar_to_default": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "torque_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "joint_vel_penalty" : torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "out_of_range_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "collision_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),

            "reward_survival" : torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "Reward_Per_Environment": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
        }

    def _setup_scene(self, render):
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(
                dt=self.config.dt,
                substeps=1,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                # max_collision_pairs=20,
            ),
            show_viewer=render)
        
        if self.terrain:
            curriculum_terrains = [
                "flat_terrain",
                # "stairs_terrain",
                # "stepping_stones_terrain",
                "fractal_terrain",
                "wave_terrain",
                "pyramid_sloped_terrain",
                "pyramid_stairs_terrain",
                "discrete_obstacles_terrain",
                "random_uniform_terrain",
            ]

            print(curriculum_terrains)
            print()
            curriculum_terrains = [[random.choice(curriculum_terrains) for _ in range(self.number_of_lanes)] for i in range(self.number_of_rows)]
            # curriculum_terrains = [[curriculum_terrains[i]] for i in range(9)]
            # self.number_of_rows = 9
            # self.number_of_lanes = 1
            if self.terrain:
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
            gs.morphs.URDF(file='//home/yayy/My/Codeeeeee/Simulators/Genesis/genesis-world/genesis/assets/urdf/go2/urdf/go2.urdf'),
        )

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx = self.robot.idx,
                link_idx_local = self.robot.get_link("base").idx_local,
                draw_debug = True
            )
        )

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

        self.scene.build(n_envs = self.config.num_envs, env_spacing = (5.0, 5.0))

    def _setup_obs_and_action_spaces(self):
        
        custom_commands_list = ["Velocity forward", "Vy Velocity", "Angular Velocity XY Plane"]
        self.custom_commands = torch.zeros((self.config.num_envs, len(custom_commands_list)), device=self.config.device)
        self.command_envs = torch.zeros(self.config.num_envs, device=self.config.device)

        self.total_obs_len = 46
        self.num_actions = 12
        self.cfg = {}
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_obs_len, ),
            dtype=float
        )
        
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_actions, ),
            dtype=float
        )

        self.actions = torch.zeros((self.config.num_envs, self.num_actions), device=self.config.device)
        self.last_actions = torch.zeros_like(self.actions, device=self.config.device)
        self.second_last_actions = torch.zeros_like(self.actions, device=self.config.device)
        self.last_dof_vel = torch.zeros((self.config.num_envs, self.num_actions), device=self.config.device)

        self.extras = dict()        
        self.extras["observations"] = dict()
        self.extras["episode"] = dict()

    def __get_linear_velocity(self):
        vel_body = transform_by_quat(self.vel_body, inv_quat(self.base_quat))
        return vel_body

    def _get_internal_info(self):
        self.joints_local_idx = []
        for links in self.robot.links[1:]:
            joints = links.joints
            for joint in joints:
                print(f"Joint name: {joint.name}, local index: {joint.dof_idx_local}")
                self.joints_local_idx.append(joint.dof_idx_local)

        self.robot.set_dofs_kp(torch.tensor([self.config.kp] * 12).to(self.config.device),dofs_idx_local = self.joints_local_idx)
        self.robot.set_dofs_kv(torch.tensor([self.config.kv] * 12).to(self.config.device),dofs_idx_local = self.joints_local_idx)

    def _get_imu_values(self):
        linear_acceleration, _angular_vel, magnetometer = self.imu.read()
        return linear_acceleration, _angular_vel

    def _get_collision_forces(self):
        collision_forces = []
        for sensor in self.contact_sensors[:-4]:
            contact_forces = sensor.read()
            collision_forces.append(contact_forces.norm(dim=-1))
        return torch.stack(collision_forces, dim=1)

    def get_local_height_of_robot(self):
        hf_np = self.random_terrain.geoms[0].metadata["height_field"]  # [H, W] int16/float
        self.hf = torch.tensor(hf_np, dtype=torch.float32, device=self.device)
        self.terrain_h_scale = self.horizontal_scale
        self.terrain_v_scale = self.vertical_scale

        hf = self.hf  # [H, W]
        hs = self.horizontal_scale
        vs = self.vertical_scale

        x_idx = self.base_x / hs
        y_idx = self.base_y / hs

        x0 = x_idx.long()
        y0 = y_idx.long()
        x1 = x0 + 1
        y1 = y0 + 1

        # clamp to valid range
        H, W = hf.shape
        x0 = x0.clamp(0, H - 1)
        y0 = y0.clamp(0, W - 1)
        x1 = x1.clamp(0, H - 1)
        y1 = y1.clamp(0, W - 1)

        # fractional position within cell
        tx = (x_idx - x0.float()).clamp(0.0, 1.0)
        ty = (y_idx - y0.float()).clamp(0.0, 1.0)

        # heights at 4 corners
        h00 = hf[x0, y0] * vs   # bottom-left
        h10 = hf[x1, y0] * vs   # bottom-right
        h01 = hf[x0, y1] * vs   # top-left
        h11 = hf[x1, y1] * vs   # top-right

        # barycentric interpolation
        # Genesis splits each cell along (x0,y0)-(x1,y1) diagonal
        upper_left = tx <= ty   # [num_envs] bool mask

        # upper-left triangle: corners (x0,y0), (x0,y1), (x1,y1)
        h_upper = h00 * (1.0 - ty) + h01 * (ty - tx) + h11 * tx

        # lower-right triangle: corners (x0,y0), (x1,y0), (x1,y1)
        h_lower = h00 * (1.0 - tx) + h10 * (tx - ty) + h11 * ty

        height = torch.where(upper_left, h_upper, h_lower)

        return self.base_z - height


    def _get_all_values_at_once(self):
        self.base_pos = self.robot.get_link("base").get_pos()
        _, self.base_ang_vel = self._get_imu_values()   #Read
        self.base_quat = self.robot.get_link("base").get_quat()  #Read
        self.euler = _quat_to_euler(self.base_quat)  #Read
        self.vel_body = self.robot.get_link("base").get_vel() 
        self.base_lin_vel_body = self.__get_linear_velocity()
        self.projected_gravity = _calculate_projected_acceleration(self.base_quat)  #Read
        self.dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx)  #Read
        self.dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)  #Read
        self.torques = self.robot.get_dofs_control_force(dofs_idx_local=self.joints_local_idx)
        self.contact_forces = self.robot.get_links_net_contact_force() #Read
        self.collision_forces = self._get_collision_forces()

        self.base_x = self.base_pos[:, 0]
        self.base_y = self.base_pos[:, 1]
        self.base_z = self.base_pos[:, 2]
        if self.terrain:
            self.base_z = self.get_local_height_of_robot()
        self.vx = self.base_lin_vel_body[:, 0]
        self.vy = self.base_lin_vel_body[:, 1]
        self.wz = self.base_ang_vel[:, 2]
        self.vz = self.base_lin_vel_body[:, 2]
        self.base_height = self.base_pos[:, 2]
        self.current_pitch = self.euler[:, 1]
        self.current_yaw = self.euler[:, 2]

    def _calculate_reward(self):

        # Rewards and Penalty for Custom Commands
        reward_for_tracking_vx = torch.exp(-torch.square(self.custom_commands[:, 0] - self.vx) / self.config.sigma_vx)
        reward_for_tracking_vy = torch.exp(-torch.square(self.custom_commands[:, 1] - self.vy) / self.config.sigma_vx)
        reward_for_tracking_wz = torch.exp(-torch.square(self.custom_commands[:, 2] - self.wz) / self.config.sigma_wz)
        height_penalty = torch.square(self.base_height - self.config.height_range)
        pitch_penalty = torch.square(self.current_pitch)
        # yaw_penalty = torch.square(self.current_yaw - self.custom_commands[:, 1])
        # Additional Penaties
        hip_deviation_penalty = torch.sum(torch.square(self.dof_pos[:, :4]), dim=1)
        lin_vel_z_penalty = torch.square(self.vz)
        roll_pitch_velocity_penalty = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        action_rate_penalty = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        second_order_action_rate = torch.sum(torch.square(self.second_last_actions - 2 * self.last_actions + self.actions), dim=1)
        reward_similar_to_default = torch.sum(torch.square(self.dof_pos - self.config.initial_positions), dim=1)
        torque_penalty = torch.sum(torch.square(self.torques), dim=1)
        joint_vel_penalty = torch.sum(torch.square(self.dof_vel), dim=1)
        out_of_range_penalty = torch.sum((self.dof_pos < self.config.URDF_specified_joint_limits[:, 0]) | (self.dof_pos > self.config.URDF_specified_joint_limits[:, 1]), dim=1)
        collision_penalty = (self.collision_forces > 1.0).any(dim=-1)
        
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

        # print(dir(self.random_terrain))
        # print(type(self.random_terrain))

        # print(dir(self.scene))
        # print(type(self.scene))

        reward_for_tracking_vx = + 3.0 * reward_for_tracking_vx
        reward_for_tracking_wz = + 2 * reward_for_tracking_wz
        reward_for_tracking_vy = + 3.0 * reward_for_tracking_vy
        # vy_penalty = - 0.5 * torch.square(self.vy)
        height_penalty = - 10.0 * height_penalty
        pitch_penalty = - 5.0 * pitch_penalty
        # yaw_penalty = - 2.0 * yaw_penalty
        hip_deviation_penalty = - 4.0 * hip_deviation_penalty
        lin_vel_z_penalty = - 0.02 * lin_vel_z_penalty
        roll_pitch_velocity_penalty = - 0.001 * roll_pitch_velocity_penalty
        action_rate_penalty = - 0.01 * action_rate_penalty
        second_order_action_rate = - 0.01 * second_order_action_rate
        reward_similar_to_default = - 0.01 * reward_similar_to_default
        torque_penalty = - 0.001 * torque_penalty
        joint_vel_penalty = - 0.005 * joint_vel_penalty
        out_of_range_penalty = - 1.0 * out_of_range_penalty
        collision_penalty = - 1.0 * collision_penalty

        reward_survival = torch.ones_like(self.base_height) * 1.0

        reward = (
            reward_for_tracking_vx
            + reward_for_tracking_wz
            + reward_for_tracking_vy
            # + vy_penalty
            + height_penalty
            + pitch_penalty
            + hip_deviation_penalty
            + lin_vel_z_penalty
            + roll_pitch_velocity_penalty
            + action_rate_penalty
            + second_order_action_rate
            + reward_similar_to_default
            + torque_penalty
            + joint_vel_penalty
            + out_of_range_penalty
            + reward_survival
        )


        terminated = ((torch.abs(self.euler[:, 0]) > 1) | (torch.abs(self.euler[:, 1]) > 1))
        out_of_trajectory = False
        if self.terrain:
            out_of_trajectory = (self.base_x < 0.5) | (self.base_x > self.terrain_length-0.5) | (self.base_y < 0.5) | (self.base_y > self.terrain_breadth - 0.5)
            # print(self.base_x, self.base_y, self.terrain_length, self.terrain_breadth)
            # print("here")
        reward[terminated] -= 500

        
        self.episode_sums["reward_for_tracking_vx"] += reward_for_tracking_vx
        self.episode_sums["reward_for_tracking_wz"] += reward_for_tracking_wz
        self.episode_sums["reward_for_tracking_vy"] += reward_for_tracking_vy
        # self.episode_sums["vy_penalty"] += vy_penalty
        self.episode_sums["height_penalty"] += height_penalty
        self.episode_sums["pitch_penalty"] += pitch_penalty
        self.episode_sums["hip_deviation_penalty"] += hip_deviation_penalty
        self.episode_sums["lin_vel_z_penalty"] += lin_vel_z_penalty
        self.episode_sums["roll_pitch_velocity_penalty"] += roll_pitch_velocity_penalty
        self.episode_sums["action_rate_penalty"] += action_rate_penalty
        self.episode_sums["second_order_action_rate"] += second_order_action_rate
        self.episode_sums["similar_to_default"] += reward_similar_to_default
        self.episode_sums["torque_penalty"] += torque_penalty
        self.episode_sums["joint_vel_penalty"] += joint_vel_penalty
        self.episode_sums["out_of_range_penalty"] += out_of_range_penalty
        self.episode_sums["collision_penalty"] += collision_penalty
        self.episode_sums["reward_survival"] += reward_survival
        self.episode_sums["Reward_Per_Environment"] += reward * 1.0
        reward *= self.config.dt

        return reward, terminated, out_of_trajectory

    def get_observations(self):
        self._get_all_values_at_once()
        obs_buf = self._get_obs()
        return TensorDict(
        {
            "policy": obs_buf["policy"],
        },
        batch_size=self.config.num_envs, device=self.config.device)

    def get_privileged_observations(self):
        return None

    def _get_obs(self):
        
        dof_pos = self.dof_pos - self.config.initial_positions
        scaled_commands = torch.stack([
            self.custom_commands[:, 0] * 1.0,   
            self.custom_commands[:, 1] * 1.0,
            self.custom_commands[:, 2] * 1.0
        ], dim=1) #scal

        # Adding noise here directly
        if self.randomize:
            base_lin_vel_body = self.base_lin_vel_body + torch.randn_like(self.base_lin_vel_body) * 0.1
            base_ang_vel = self.base_ang_vel + torch.randn_like(self.base_ang_vel) * 0.05
            projected_gravity = self.projected_gravity + torch.randn_like(self.projected_gravity) * 0.02
            euler = self.euler + torch.randn_like(self.euler) * 0.05
            noisy_dof_pos = dof_pos + torch.randn_like(dof_pos) * 0.01
            dof_vel = self.dof_vel + torch.randn_like(self.dof_vel) * 0.5

        else:
            base_lin_vel_body = self.base_lin_vel_body
            base_ang_vel = self.base_ang_vel
            projected_gravity = self.projected_gravity
            euler = self.euler
            noisy_dof_pos = dof_pos
            dof_vel = self.dof_vel

        current_obs = TensorDict({          
            "policy": torch.cat([
            # base_lin_vel_body,                        # 3 
            base_ang_vel * 0.5,                       # 3
            projected_gravity ,                            # 3
            euler[:, 2].unsqueeze(1),                                # 1
            noisy_dof_pos,                                       # 12
            dof_vel * 0.05,                                # 12
            self.actions,                                  # 12
            scaled_commands,                               # 2
            ], dim=1),
        }, batch_size=self.config.num_envs, device=self.config.device)

        return current_obs

    def _reset_idx(self, envs_idx):
        
        if len(envs_idx) == 0:
            return
        
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            avg_reward = torch.mean(value[envs_idx]) / self.config.max_episode_length
            self.extras["episode"]["rew_" + key] = avg_reward
            
            value[envs_idx] = 0.0

        self.robot.set_dofs_position(
            self.config.initial_positions.repeat(len(envs_idx), 1),
            dofs_idx_local=self.joints_local_idx,
            envs_idx=envs_idx
        )

        self.robot.set_dofs_velocity(
            torch.zeros((len(envs_idx), 12), device=self.config.device),
            dofs_idx_local=self.joints_local_idx,
            envs_idx=envs_idx
        )

        respawn_pos = torch.zeros((len(envs_idx), 3), device=self.device)
        if self.terrain:
            respawn_pos[:, 0] = self.respawn_x_range[0] + torch.rand(len(envs_idx), device=self.device) * (self.respawn_x_range[1] - self.respawn_x_range[0])
            respawn_pos[:, 1] = self.respawn_y_range[0] + torch.rand(len(envs_idx), device=self.device) * (self.respawn_y_range[1] - self.respawn_x_range[0])
            respawn_pos[:, 2] = self.respawn_z_range[0] + torch.rand(len(envs_idx), device=self.device) * (self.respawn_z_range[1] - self.respawn_z_range[0])
        else:
            respawn_pos[:, 0] = -30 + torch.rand(len(envs_idx), device=self.device) * 60
            respawn_pos[:, 1] = -30 + torch.rand(len(envs_idx), device=self.device) * 60
            respawn_pos[:, 2] = 0.25 + torch.rand(len(envs_idx), device=self.device) * 0.17
        self.episode_length_buf[envs_idx] = 0
        self.actions[envs_idx] = 0
        self.last_actions[envs_idx] = 0
        self.second_last_actions[envs_idx] = 0
        self.last_dof_vel[envs_idx] = 0

        
        base_quat = torch.tensor([1, 0, 0, 0], device=self.config.device)
        self.robot.set_quat(quat=base_quat, envs_idx=envs_idx)
        self.robot.set_pos(respawn_pos, envs_idx=envs_idx)
        if self.randomize:
            self.domainrandomizer.randomize(envs_idx)
            self.domainrandomizer.action_buffer[envs_idx] = 0
        self.update_commands(envs_idx)

   

    def reset(self):
        print("Resetting Environment")
        all_idx = torch.arange( self.config.num_envs, device=self.config.device)
        self._reset_idx(all_idx)
        self.scene.reset()
        self._get_all_values_at_once()
        self.obs_buf = self._get_obs()
        print("Environment Reset")
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

        target_dof_pos = self.config.initial_positions + action * self.config.action_scale
        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local = self.joints_local_idx)
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

        # self.previous_joint_velocity = self.robot.get_dofs_velocity(dofs_idx_local = self.joints_local_idx)
        self.previous_joint_velocity = self.dof_vel.clone()

        reset_ids = torch.nonzero(dones).flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        observation = self._get_obs()
        
        info = {
            "final_obs":   final_obs["policy"].detach().cpu().numpy(),
            "terminated":  terminated.detach().cpu().numpy(),
            "robot_pos":    self.base_pos[0],
            "robot_quat":   self.base_quat[0],
        }

        return observation, reward, dones, self.extras   

