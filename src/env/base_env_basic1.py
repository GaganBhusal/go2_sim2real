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

class WalkENV(gym.Env):

    def __init__(self, config, render = False, backend = gs.gpu, num_envs = 100, device="cuda", plane = True, t_x = 20, t_y = 50, number_of_lanes=1, number_of_rows=10):
        super().__init__()

        self.config = config
        self.plane = plane
        self.num_envs = self.config.num_envs
        self.device = self.config.device
        self.episode_length_buf = torch.zeros(self.config.num_envs, dtype=torch.int, device=self.config.device)

        self._setup_scene(render)
        self._get_internal_info()
        self._setup_obs_and_action_spaces()

        # self.domainrandomizer = DomainRandomization(self.robot, self.num_envs, self.joints_local_idx)



        self.episode_sums = {
            "reward_for_tracking_vx": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_for_tracking_wz": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "vy_penalty": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
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
        
        if self.plane:
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
        
        custom_commands_list = ["Velocity forward", "Angular Velocity XY Plane"]
        self.custom_commands = torch.zeros((self.config.num_envs, len(custom_commands_list)), device=self.config.device)
        self.command_envs = torch.zeros(self.config.num_envs, device=self.config.device)

        self.total_obs_len = 48
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


    def _get_all_values_at_once(self):
        self.base_pos = self.robot.get_link("base").get_pos()
        _, self.base_ang_vel = self._get_imu_values()
        self.base_quat = self.robot.get_link("base").get_quat()
        self.euler = _quat_to_euler(self.base_quat)
        self.vel_body = self.robot.get_link("base").get_vel()
        self.base_lin_vel_body = self.__get_linear_velocity()
        self.projected_gravity = _calculate_projected_acceleration(self.base_quat)
        self.dof_pos = self.robot.get_dofs_position(dofs_idx_local=self.joints_local_idx)
        self.dof_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.joints_local_idx)
        self.torques = self.robot.get_dofs_control_force(dofs_idx_local=self.joints_local_idx)
        self.contact_forces = self.robot.get_links_net_contact_force()
        self.collision_forces = self._get_collision_forces()

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
        reward_for_tracking_wz = torch.exp(-torch.square(self.custom_commands[:, 1] - self.wz) / self.config.sigma_wz)
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



        reward_for_tracking_vx = + 3.0 * reward_for_tracking_vx
        reward_for_tracking_wz = + 2 * reward_for_tracking_wz
        vy_penalty = - 0.5 * torch.square(self.vy)
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
            + vy_penalty
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


        terminated = ((torch.abs(self.euler[:, 0]) > 0.26) | (torch.abs(self.euler[:, 1]) > 0.26) | (self.base_height < 0.18))
        out_of_trajectory = False
        reward[terminated] -= 500

        
        self.episode_sums["reward_for_tracking_vx"] += reward_for_tracking_vx
        self.episode_sums["reward_for_tracking_wz"] += reward_for_tracking_wz
        self.episode_sums["vy_penalty"] += vy_penalty
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
        ], dim=1) #scal


        current_obs = TensorDict({          
            "policy": torch.cat([
            self.base_lin_vel_body,                        # 3 
            self.base_ang_vel * 0.5,                       # 3
            self.projected_gravity ,                            # 3
            self.euler[:, 2].unsqueeze(1),                                # 1
            dof_pos,                                       # 12
            self.dof_vel * 0.05,                                # 12
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

        self.episode_length_buf[envs_idx] = 0
        self.actions[envs_idx] = 0
        self.last_actions[envs_idx] = 0
        self.second_last_actions[envs_idx] = 0
        self.last_dof_vel[envs_idx] = 0

        # y = torch.randint(low=20, high=self.terrain_breadth-20, size=(len(envs_idx), ))
        x = 0
        y = 0
        z = 0.4


        base_quat = torch.tensor([1, 0, 0, 0], device=self.config.device)
        self.robot.set_quat(quat=base_quat, envs_idx=envs_idx)
        base_pos = torch.tensor([x, y, z], device=self.config.device)
        self.robot.set_pos(base_pos, envs_idx=envs_idx)
        # self.domainrandomizer.randomize(envs_idx)

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
        self.custom_commands[idx, 1] = torch.rand(len(idx), device=self.config.device) * (self.config.wz_range[1] - self.config.wz_range[0]) + self.config.wz_range[0]
        self.custom_commands[idx[torch.rand(len(idx), device=self.config.device) < 0.25]] = 0.0
        
    def step(self, action):

        self.actions = action.clone()

        target_dof_pos = self.config.initial_positions + action * self.config.action_scale
        self.robot.control_dofs_position(target_dof_pos, dofs_idx_local = self.joints_local_idx)
        self.scene.step()

        self._get_all_values_at_once()
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

