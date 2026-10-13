import torch
from tensordict import TensorDict
from .base_env import BaseQuadrupedEnv
from .rewards import RewardFunctions

class QuadrupedWalk(BaseQuadrupedEnv, RewardFunctions):

    def _init_episode_sums(self):
        return {
            "reward_for_tracking_vx": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_for_tracking_wz": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),            
            "reward_for_tracking_vy": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_lin_vel_z" : torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_base_height": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_torques" : torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_dof_vel": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_dof_acc": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_action_rate": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_feet_air_time": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_stand_still" : torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_base_roll_pitch_rates": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_dof_pos_limits_soft": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_contact": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_feet_swing_height": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_survival" : torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_collision": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_hip_deviation": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_similar_to_default": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_feet_slip": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "reward_foot_clearance": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
            "Reward_Per_Environment": torch.zeros(self.config.num_envs, dtype=torch.float, device=self.config.device),
        }

    def _calculate_reward(self):

        reward_for_tracking_vx = self.config.reward_scales_tracking_vx * self._reward_for_tracking_vx()
        reward_for_tracking_vy = self.config.reward_scales_tracking_vy * self._reward_for_tracking_vy()
        reward_for_tracking_wz = self.config.reward_scales_tracking_wz * self._reward_for_tracking_wz()
        reward_lin_vel_z = self.config.reward_scales_lin_vel_z * self._reward_lin_vel_z()
        reward_base_height = self.config.reward_scales_base_height * self._reward_base_height()
        reward_torques = self.config.reward_scales_torques * self._reward_torques()
        reward_dof_vel = self.config.reward_scales_dof_vel * self._reward_dof_vel()
        reward_dof_acc = self.config.reward_scales_dof_acc * self._reward_dof_acc()
        reward_action_rate = self.config.reward_scales_action_rate * self._reward_action_rate()
        reward_feet_air_time = self.config.reward_scales_feet_air_time * self._reward_feet_air_time()
        reward_stand_still = self.config.reward_scales_stand_still * self._reward_stand_still()
        reward_base_roll_pitch_rates = self.config.reward_scales_base_roll_pitch_rates * self._reward_base_roll_pitch_rates()
        reward_dof_pos_limits_soft = self.config.reward_scales_dof_pos_limits_soft * self._reward_dof_pos_limits_soft()
        reward_contact = self.config.reward_scales_contact * self._reward_contact()
        reward_feet_swing_height = self.config.reward_scales_feet_swing_height * self._reward_feet_swing_height()
        reward_survival = torch.ones_like(self.base_height) * self.config.reward_scales_survival
        reward_termination = self._reward_termination()

        reward_collision = self.config.reward_scales_collision * self._reward_collision()
        reward_hip_deviation = self.config.reward_scales_hip_deviation_penalty * self._hip_deviation_penalty()
        reward_similar_to_default = self.config.reward_scales_dof_pos * self._reward_similar_to_default()
        reward_feet_slip = self.config.reward_scales_foot_slip * self._reward_feet_slip()

        reward_foot_clearance = self.config.reward_scales_foot_clearance * self._reward_feet_clearance()

        reward = (
            reward_for_tracking_vx
            + reward_for_tracking_wz
            + reward_for_tracking_vy
            + reward_lin_vel_z
            + reward_base_height
            + reward_torques
            + reward_dof_vel
            + reward_dof_acc
            + reward_action_rate
            + reward_feet_air_time
            + reward_stand_still
            + reward_base_roll_pitch_rates
            + reward_dof_pos_limits_soft
            + reward_contact
            + reward_feet_swing_height
            + reward_survival
            + reward_collision
            + reward_hip_deviation
            + reward_similar_to_default
            + reward_foot_clearance
            + reward_feet_slip
        )
        reward[reward_termination] += self.config.reward_scales_termination_penalty


        self.episode_sums["reward_for_tracking_vx"] += reward_for_tracking_vx
        self.episode_sums["reward_for_tracking_wz"] += reward_for_tracking_wz
        self.episode_sums["reward_for_tracking_vy"] += reward_for_tracking_vy
        self.episode_sums["reward_lin_vel_z"] += reward_lin_vel_z
        self.episode_sums["reward_base_height"] += reward_base_height
        self.episode_sums["reward_collision"] += reward_collision
        self.episode_sums["reward_hip_deviation"] += reward_hip_deviation
        self.episode_sums["reward_similar_to_default"] += reward_similar_to_default
        self.episode_sums["reward_torques"] += reward_torques
        self.episode_sums["reward_feet_slip"] += reward_feet_slip
        self.episode_sums["reward_dof_vel"] += reward_dof_vel
        self.episode_sums["reward_dof_acc"] += reward_dof_acc
        self.episode_sums["reward_action_rate"] += reward_action_rate
        self.episode_sums["reward_feet_air_time"] += reward_feet_air_time
        self.episode_sums["reward_stand_still"] += reward_stand_still
        self.episode_sums["reward_base_roll_pitch_rates"] += reward_base_roll_pitch_rates
        self.episode_sums["reward_dof_pos_limits_soft"] += reward_dof_pos_limits_soft
        self.episode_sums["reward_contact"] += reward_contact
        self.episode_sums["reward_feet_swing_height"] += reward_feet_swing_height
        self.episode_sums["reward_survival"] += reward_survival
        self.episode_sums["reward_foot_clearance"] += reward_foot_clearance
        self.episode_sums["Reward_Per_Environment"] += reward * 1.0
        
        reward *= self.config.dt

        return reward, reward_termination, False

    def _get_obs(self):
        dof_pos = self.dof_pos - self.config.initial_positions

        scaled_commands = torch.stack([
            self.custom_commands[:, 0] * 1.0,   
            self.custom_commands[:, 1] * 1.0,
            self.custom_commands[:, 2] * 1.0
        ], dim=1)

        # # Adding noise scaled with DR curriculum
        # if self.randomize:
        #     scale = self.dr_scale
        #     base_lin_vel_body = self.base_lin_vel_body + torch.randn_like(self.base_lin_vel_body) * 0.1 * scale
        #     base_ang_vel = self.base_ang_vel + torch.randn_like(self.base_ang_vel) * 0.05 * scale
        #     projected_gravity = self.projected_gravity + torch.randn_like(self.projected_gravity) * 0.02 * scale
        #     euler = self.euler + torch.randn_like(self.euler) * 0.05 * scale
        #     noisy_dof_pos = dof_pos + torch.randn_like(dof_pos) * 0.01 * scale
        #     dof_vel = self.dof_vel + torch.randn_like(self.dof_vel) * 0.5 * scale
        # else:
        #     base_lin_vel_body = self.base_lin_vel_body
        #     base_ang_vel = self.base_ang_vel
        #     projected_gravity = self.projected_gravity
        #     euler = self.euler
        #     noisy_dof_pos = dof_pos
        #     dof_vel = self.dof_vel


        # current_obs = TensorDict({          
        #     "policy": torch.cat([
        #     base_ang_vel * 0.5,                       
        #     projected_gravity ,                            
        #     noisy_dof_pos,                                       
        #     dof_vel * 0.05,                                
        #     self.actions,                                  
        #     scaled_commands, 
        #     self.sine_phase.unsqueeze(1),
        #     self.cosine_phase.unsqueeze(1)                             
        #     ], dim=1),

        #     "privileged": torch.cat([
        #     base_lin_vel_body * 2.0,                       
        #     base_ang_vel * 0.5,                       
        #     self.base_z.unsqueeze(1),                     
        #     projected_gravity ,                            
        #     noisy_dof_pos,                                       
        #     dof_vel * 0.05,                                
        #     scaled_commands, 
        #     self.feet_air_time,                           
        #     self.actions,                                  
        #     self.sine_phase.unsqueeze(1),
        #     self.cosine_phase.unsqueeze(1)
        #     ], dim=1),
        # }, batch_size=self.config.num_envs, device=self.config.device)

        clean_lin_vel = self.base_lin_vel_body
        clean_ang_vel = self.base_ang_vel
        clean_gravity = self.projected_gravity
        clean_dof_pos = dof_pos
        clean_dof_vel = self.dof_vel

        if self.randomize:
            scale = getattr(self, 'dr_scale', 1.0)
            actor_ang_vel = clean_ang_vel + torch.randn_like(clean_ang_vel) * 0.05 * scale
            actor_gravity = clean_gravity + torch.randn_like(clean_gravity) * 0.02 * scale
            actor_dof_pos = clean_dof_pos + torch.randn_like(clean_dof_pos) * 0.01 * scale
            actor_dof_vel = clean_dof_vel + torch.randn_like(clean_dof_vel) * 0.5 * scale
        else:
            actor_ang_vel = clean_ang_vel
            actor_gravity = clean_gravity
            actor_dof_pos = clean_dof_pos
            actor_dof_vel = clean_dof_vel

        current_obs = TensorDict({          
            "policy": torch.cat([
                actor_ang_vel * 0.5,                       
                actor_gravity,                            
                actor_dof_pos,                                       
                actor_dof_vel * 0.05,                                
                self.actions,                                  
                scaled_commands, 
                self.sine_phase.unsqueeze(1),
                self.cosine_phase.unsqueeze(1)                             
            ], dim=1),

            "privileged": torch.cat([
                clean_lin_vel * 2.0,                       
                clean_ang_vel * 0.5,                       
                self.base_z.unsqueeze(1),                     
                clean_gravity,                            
                clean_dof_pos,                                       
                clean_dof_vel * 0.05,                                
                scaled_commands, 
                self.feet_air_time,                           
                self.actions,                                  
                self.sine_phase.unsqueeze(1),
                self.cosine_phase.unsqueeze(1)
            ], dim=1),
        }, batch_size=self.config.num_envs, device=self.config.device)

        return current_obs
