import torch
from dataclasses import dataclass

@dataclass
class TaskCfg:
    # For commands
    vx_range: tuple = (-1.0, 1.0)
    vy_range: tuple = (-1.0, 1.0)
    wz_range: tuple = (-1.0, 1.0)
    
    height_range: float = 0.31
    pitch_range: tuple = (-0.15, 0.15)
    roll_range: tuple = (-0.15, 0.15)
    yaw_range: tuple = (-0.15, 0.15)

    sigma_vx: float = 0.25
    sigma_wz: float = 0.25

    target_air_time: float = 0.15
    trot_offsets: tuple = (0.0, 0.5, 0.5, 0.0)
    step_freq: float = 2.0
    min_step_freq: float = 0.0
    freq_speed_scale: float = 1.0
    target_foot_height: float = 0.10

    gait_period = 0.5
    gait_offset = [0.0, 0.5, 0.5, 0.0]



# FOR FIRST ITERATION I AM GIVING TORQUE AND VELOCITY AND ACCELERATION PENALTY LOW AND IN SECOND ITERATION I WILL INCREASE IT MORE A BIT TO LET IT TO EXPLORE
    # reward_scales_torques: float = -0.000001 
    # reward_scales_dof_vel: float = -1.0e-4
    # reward_scales_dof_acc: float = -5.0e-7
    # reward_scales_action_rate: float = -0.001


    reward_scales_tracking_vx: float = 1.0
    reward_scales_tracking_vy: float = 1.0
    reward_scales_tracking_wz: float = 1.0
    reward_scales_lin_vel_z: float = -2.0
    reward_scales_base_height: float = -30.0

    reward_scales_torques: float = -0.00001
    reward_scales_dof_vel: float = -1.0e-3
    reward_scales_dof_acc: float = -5.0e-7
    reward_scales_action_rate: float = -0.01

    reward_scales_collision: float = -1.0
    reward_scales_dof_pos_limit: float = -1.0
    reward_scales_base_roll_pitch_rates: float = -0.05
    reward_scales_dof_pos_limits_soft: float = -0.1

    reward_scales_feet_air_time: float = 1.5
    reward_scales_stand_still: float = -0.4
    reward_scales_contact: float = 0.5
    reward_scales_feet_swing_height: float = 0.5

    reward_scales_survival: float = 1.0
    reward_scales_termination_penalty: float = -500.0

    reward_scales_hip_deviation_penalty: float = -2.0
    reward_scales_dof_pos: float = -0.5
    reward_scales_foot_slip: float = -0.25

    reward_scales_foot_clearance: float = 0.0
















