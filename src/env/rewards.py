import torch
from mjlab.utils.lab_api.math import quat_mul, quat_conjugate, axis_angle_from_quat
from genesis.utils.geom import transform_by_quat, inv_quat

class RewardFunctions:
    
    def _reward_for_tracking_vx(self):
        return torch.exp(-torch.square(self.custom_commands[:, 0] - self.vx) / self.config.sigma_vx)
    
    def _reward_for_tracking_vy(self):
        return torch.exp(-torch.square(self.custom_commands[:, 1] - self.vy) / self.config.sigma_vx)
    
    def _reward_for_tracking_wz(self):
        return torch.exp(-torch.square(self.custom_commands[:, 2] - self.wz) / self.config.sigma_wz)

    def _reward_upright_exp(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim = 1)
    
    def _reward_lin_vel_z(self):
        return torch.square(self.vz)
    
    def _reward_base_height(self):
        return torch.square(self.base_height - self.config.height_range)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel)/self.config.dt), dim=1)
    
    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_second_order_action_rate(self):
        return torch.sum(torch.square(self.second_last_actions - 2 * self.last_actions + self.actions), dim=1)
    
    def _reward_termination(self):
        return (torch.abs(self.euler[:, 0]) > 1.0) | (torch.abs(self.euler[:, 1]) > 1.0) | (self.base_height < 0.18)

    def _reward_feet_air_time(self):
        contact = self.feet_contacts
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact.clone()

        is_swing = self.leg_phase >= 0.55

        # Only accumulate air time during swing phase
        self.feet_air_time += self.config.dt * is_swing.float()

        first_contact = (self.feet_air_time > 0.) * contact_filt
        air_time_diff = self.feet_air_time - self.config.target_air_time
        rew_air_time = torch.sum(air_time_diff * first_contact.float(), dim=1)

        total_cmd = torch.norm(self.custom_commands[:, :2], dim=1) + torch.abs(self.custom_commands[:, 2])
        rew_air_time *= (total_cmd > 0.1).float()

        self.feet_air_time *= (~contact_filt)
        return rew_air_time

    def _reward_stand_still(self):
        # Deflection from initial position * linear velocity * angular_velocity
        return torch.sum(torch.abs(self.dof_pos - self.config.initial_positions), dim=1) * (torch.norm(self.custom_commands[:, :2], dim=1) < 0.1) * (torch.abs(self.custom_commands[:, 2]) < 0.1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.square(self.dof_pos - self.config.initial_positions), dim=1)

    def _reward_base_roll_pitch_rates(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_pos_limits_soft(self):
        limit_min = self.config.clamped_joint_limits[:, 0]
        limit_max = self.config.clamped_joint_limits[:, 1]
        margin = 0.1 * (limit_max - limit_min)
        
        penalty = torch.sum(
            torch.square(torch.clamp(margin - (self.dof_pos - limit_min), min=0.0)) + 
            torch.square(torch.clamp(margin - (limit_max - self.dof_pos), min=0.0)), 
            dim=1
        )
        return penalty
    

    # def _reward_feet_swing_height(self):

    #     h_max = self.config.target_foot_height
    #     swing_phase = (self.leg_phase - 0.55) / 0.45
    #     swing_phase = torch.clamp(swing_phase, 0.0, 1.0)
    #     target_height = h_max * torch.square(torch.sin(torch.pi * swing_phase))
        
    #     swing_mask = self.leg_phase >= 0.55
    #     target_height = torch.where(swing_mask, target_height, torch.zeros_like(target_height))
        
    #     # contact = self.feet_contacts
    #     pos_error = torch.square((self.feet_pos[:, :, 2] - self.config.LEG_OFFSET) - target_height) * swing_mask
    #     return torch.sum(pos_error, dim=1)  
    
    def _reward_pitch_penalty(self):
        return torch.exp(-torch.square(self.current_pitch)/self.config.sigma_pitch)
        
    def _reward_roll_penalty(self):
        return torch.exp(-torch.square(self.current_roll)/self.config.sigma_roll)

    def _reward_collision(self):
        # return torch.sum(self.whole_body_contacts[:, torch.cat([self.config.HIPS - 1, self.config.THIGHS-1])])
        return 0

    def _hip_deviation_penalty(self):
        return torch.sum(torch.square(self.dof_pos[:, :4]), dim=1)

    def _reward_contact(self):
        is_stance = self.leg_phase < 0.55
        contact = self.feet_contacts.bool()
        match = ~(contact ^ is_stance)
        reward = match.float().mean(dim=1)   

        linear_norm = torch.norm(self.custom_commands[:, :2], dim=1)
        angular_norm = torch.abs(self.custom_commands[:, 2])
        total_command = linear_norm + angular_norm
        reward *= (total_command > 0.1).float()

        return reward
    
    # def _reward_feet_air_time(self):
    #     threshold = self.config.target_air_time  # Required Feet Air Time 

    #     contact = self.feet_contacts   # (num_envs, 4)  Check whether these 4 foots are in ground or in air
    #     contact_filt = torch.logical_or(contact, self.last_contacts)  #Does similar making that check more roboust
    #     self.last_contacts = contact.clone() 

    #     self.feet_air_time += self.config.dt       #Feet air time of all 4 legs
    #     self.feet_contact_time += self.config.dt   #Contact time of all 4 legs

    #     in_contact = contact_filt  #This is just in_contact means the foot is in ground contact
    #     in_mode_time = torch.where(
    #         in_contact, 
    #         self.feet_contact_time, 
    #         self.feet_air_time
    #         )  #IF the leg is in ground contact in_mode_time is contact time else feet air time


    #     single_stance = torch.mean(in_contact.float(), dim=1) == 0.5  #Two out of 4 legs should be in contact or vice versa

    #     # bottleneck across legs -> forces symmetry, not just per-leg sum
    #     mode_time = torch.min(
    #         torch.where(
    #             single_stance.unsqueeze(-1), 
    #             in_mode_time, 
    #             torch.zeros_like(in_mode_time)
    #             ), 
    #         dim=1
    #     )[0] #Minimum time when there is single_stance and select time according to mode


    #     error = torch.abs(mode_time - threshold) # Error
    #     rew_air_time = torch.clamp(threshold - error, min=0.0, max = 0.4) # Clamping

    #     # command gating: linear + angular, not just linear
    #     linear_norm = torch.norm(self.custom_commands[:, :2], dim=1)
    #     angular_norm = torch.abs(self.custom_commands[:, 2])
    #     total_command = linear_norm + angular_norm
    #     rew_air_time *= (total_command > 0.1).float()  # Commands should be greater than 0.1

    #     self.feet_air_time *= (~contact_filt)
    #     self.feet_contact_time *= contact_filt

    #     return rew_air_time



    # def _reward_feet_swing_height(self):
    #     target_height = self.config.target_foot_height  # Target Foot Height

    #     contact = self.feet_contacts   # (num_envs, 4)
    #     in_air = ~contact

    #     foot_heights = torch.clamp(self.feet_pos[:, :, 2] - self.config.LEG_OFFSET, min = 0.0)  #Foor Height relative to ground offset

    #     self.peak_heights = torch.where(
    #         in_air,
    #         torch.maximum(self.peak_heights, foot_heights),
    #         self.peak_heights,
    #     )  # Peak Height reached during in air

    #     first_contact = contact & (self.peak_heights > 0.0)

    #     linear_norm = torch.norm(self.custom_commands[:, :2], dim=1)
    #     angular_norm = torch.abs(self.custom_commands[:, 2])
    #     total_command = linear_norm + angular_norm
    #     active = (total_command > 0.1).float()

    #     error = self.peak_heights / target_height - 1.0
    #     cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active

    #     # reset peak tracker for legs that just landed
    #     self.peak_heights = torch.where(
    #         first_contact,
    #         torch.zeros_like(self.peak_heights),
    #         self.peak_heights,
    #     )

    #     return cost

    def _reward_feet_slip(self):
        contact = self.feet_contacts.bool()   

        foot_vel_xy = self.feet_vel[:, :, :2]       
        vel_xy_norm_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)   

        cost = torch.sum(vel_xy_norm_sq * contact.float(), dim=1)

        linear_norm = torch.norm(self.custom_commands[:, :2], dim=1)
        angular_norm = torch.abs(self.custom_commands[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > 0.01).float()   

        return cost * active

    def _reward_feet_clearance(self):
        # target_height = self.config.target_foot_height
        # in_swing = (~self.feet_contacts).float()                        # (num_envs, 4)

        # foot_z = self.feet_pos[:, :, 2] - self.config.LEG_OFFSET          # ground-relative height
        # foot_vel_xy_norm = torch.norm(self.feet_vel[:, :, :2], dim=-1)    # horizontal foot speed

        # height_error = torch.square(target_height - foot_z)
        # penalty = torch.sum(in_swing * height_error * foot_vel_xy_norm, dim=1)

        # linear_norm = torch.norm(self.custom_commands[:, :2], dim=1)
        # angular_norm = torch.abs(self.custom_commands[:, 2])    
        # total_command = linear_norm + angular_norm
        # active = (total_command > 0.1).float()

        # return penalty * active

        return 0

    def _reward_feet_swing_height(self):
        h_max = self.config.target_foot_height # 0.10m
        
        # Map leg_phase [0.5, 1.0] -> swing_phase [0.0, 1.0]
        swing_phase = torch.clamp((self.leg_phase - 0.55) / 0.45, 0.0, 1.0)
        
        # Parabolic target height arc: highest in the middle of swing (s = 0.5)
        target_height = h_max * torch.square(torch.sin(torch.pi * swing_phase))
        
        is_swing = self.leg_phase >= 0.55
        in_air = ~self.feet_contacts.bool()
        
        # Ground-relative foot tip height
        if self.terrain:
            foot_heights = self.feet_pos[:, :, 2] - (self.base_pos[:, 2] - self.base_height)
        else:
            foot_heights = self.feet_pos[:, :, 2]
        
        foot_heights = torch.clamp(foot_heights, min=0.0)
        
        # Exponential tracking reward: 1.0 per leg when perfectly matching target curve!
        height_diff = foot_heights - target_height
        reward = torch.exp(-torch.square(height_diff) / 0.005) * is_swing.float() * in_air.float()
        
        cmd_norm = torch.norm(self.custom_commands[:, :2], dim=1) + torch.abs(self.custom_commands[:, 2])
        return torch.sum(reward, dim=1) * (cmd_norm > 0.1).float()
