import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
import pygame
import keyboard

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))
from configs.env_configs import WalkEnvCfg

from unitree_sdk2py.core.channel import ChannelFactory, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient



class ActorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(47, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12)
        )
        self.register_buffer("obs_mean", torch.zeros(1, 47))
        self.register_buffer("obs_std", torch.ones(1, 47))

    def forward(self, obs):
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return self.mlp(obs)

class Custom:
    def __init__(self, config, policy):

        self.config = config
        self.actions = torch.zeros(12, dtype=torch.float32).to(self.config.device)
        
        self.POLICY_EVERY_N = 10 
        self.activate_joint_limits = False
        self.Kp = 40
        self.Kd = 4.9
        self.gait_period = 0.5
        self.stand_percent = 0.0
        self.dt = 0.002
        self.policy = policy
        self.stand_duration = 1000

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  
        self.firstRun = True
        self.emergency_stop = False

        self.startPos = torch.zeros(12, device=self.config.device)
        
        self.GO2_RANGE = torch.tensor([
            [-40, 40],
            [-40, 40],
            [-40, 40],
            [-40, 40],

            [-90, 200],
            [-90, 200],

            [-30, 260],
            [-30, 260],

            [-160, -50],
            [-160, -50],
            [-160, -50],
            [-160, -50]
        ])

        

        self.SIM2REAL = [1, 5, 9, 
                         0, 4, 8,
                         3, 7, 11,
                         2, 6, 10]
        
        self.REAL2SIM = [3, 0, 9, 6,
                         4, 1, 10, 7,
                         5, 2, 11, 8]
        
        self.joint_pos_sim_deg = torch.zeros(12, device=self.config.device)
        self.prev_actions_to_motor = self.config.initial_positions[self.SIM2REAL].clone().float()
        self.prev_actions_to_motor_fir = self.config.initial_positions[self.SIM2REAL].clone().float()
        self.actions_to_motor = self.config.initial_positions[self.SIM2REAL].clone().float()
        self.not_in_stand_position = True
        self.loop_count = 0

        self.commands = torch.zeros(3, dtype=torch.float32)  # [vx, vy, wz]

        self.MAX_VX = 0.95
        self.MAX_VY = 0.95
        self.MAX_WZ = 0.95
        self.DEADZONE = 0.1

        # Init pygame joystick
        # pygame.init()
        # pygame.joystick.init()
        # if pygame.joystick.get_count() == 0:
        #     raise RuntimeError("No controller detected!")
        # self.joystick = pygame.joystick.Joystick(0)
        # self.joystick.init()
        # print(f"Controller connected: {self.joystick.get_name()}")

        self.lowCmdWriteThreadPtr = None
        self.motor_target = self.config.initial_positions[self.SIM2REAL].clone().float()

        self.crc = CRC()

        self.file = open('robot_data.csv', 'w', newline='', encoding='utf-8')
    # Public methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.dt, target=self.MotorWrite, name="motorwrite"
        )
        self.lowCmdWriteThreadPtr.Start()

        self._run_policy_loop()

    def slew_limit(self, prev_q, new_q, max_step_rad=0.2):
        delta = torch.clamp(new_q - prev_q, -max_step_rad, max_step_rad)
        return prev_q + delta

    def _run_policy_loop(self):

        dt = 1/50.0

        next_tick = time.monotonic()

        while not self.emergency_stop:
            now = time.monotonic()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += dt

            if self.low_state is None or self.not_in_stand_position:
                continue

            self.get_actions()

    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE

        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q= go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        # print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]])
        # print("IMU state: ", msg.imu_state)
        # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)

    def recieve_controller_commands(self):
        pygame.event.pump()

        raw_vx = -self.joystick.get_axis(1)  # Left stick Y
        raw_vy = -self.joystick.get_axis(0)  # Left stick X
        raw_wz = -self.joystick.get_axis(3)  # Right stick X

        vx = raw_vx if abs(raw_vx) > self.DEADZONE else 0.0
        vy = raw_vy if abs(raw_vy) > self.DEADZONE else 0.0
        wz = raw_wz if abs(raw_wz) > self.DEADZONE else 0.0

        target_vx = vx * self.MAX_VX
        target_vy = vy * self.MAX_VY
        target_wz = wz * self.MAX_WZ

        # Low-pass filter — same as reference, prevents jerking
        self.commands[0] += (target_vx - self.commands[0]) * 0.1
        self.commands[1] += (target_vy - self.commands[1]) * 0.1
        self.commands[2] += (target_wz - self.commands[2]) * 0.1

    def recieve_controller_commands_keyboard(self):

        raw_vx = 0.0
        raw_vy = 0.0
        raw_wz = 0.0

        if keyboard.is_pressed('w'): raw_vx += 1.0
        if keyboard.is_pressed('s'): raw_vx -= 1.0
        
        if keyboard.is_pressed('a'): raw_vy += 1.0
        if keyboard.is_pressed('d'): raw_vy -= 1.0
        
        if keyboard.is_pressed('left'): raw_wz += 1.0
        if keyboard.is_pressed('right'): raw_wz -= 1.0

        target_vx = raw_vx * self.MAX_VX
        target_vy = raw_vy * self.MAX_VY
        target_wz = raw_wz * self.MAX_WZ

        # Low-pass filter — keeps the real robot from jerking
        self.commands[0] += (target_vx - self.commands[0]) * 0.04
        self.commands[1] += (target_vy - self.commands[1]) * 0.04
        self.commands[2] += (target_wz - self.commands[2]) * 0.04

    @staticmethod
    def _quat_to_euler(q):
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

    @staticmethod
    def _calculate_projected_acceleration(base_quat):
        w, x, y, z = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
        gx = 2 * (x*z - w*y)
        gy = 2 * (y*z + w*x)
        gz = 1 - 2 * (x*x + y*y)
        return torch.stack([-gx, -gy, -gz])

    def _get_obs_space(self):
        msg = self.low_state  

        gyro = torch.tensor(list(msg.imu_state.gyroscope), dtype=torch.float32).to(self.config.device)
        quat = torch.tensor(list(msg.imu_state.quaternion), dtype=torch.float32).to(self.config.device)

        joint_pos = torch.tensor([msg.motor_state[i].q  for i in range(12)], dtype=torch.float32).to(self.config.device)
        joint_vel = torch.tensor([msg.motor_state[i].dq for i in range(12)], dtype=torch.float32).to(self.config.device)        


        joint_pos = joint_pos[self.REAL2SIM]
        joint_vel = joint_vel[self.REAL2SIM]


        joint_pos = joint_pos - self.config.initial_positions
        
        # print(f"Sim Order : {to/rch.rad2deg(joint_pos[1])} \t {torch.rad2deg(joint_pos[5])} \t {torch.rad2deg(joint_pos[9])}\n\n")
        # print(f"Joint Pos (SIM order) : {torch.rad2deg(joint_pos)[0]}\tJOint Pos 2 : {torch.rad2deg(joint_pos)[6]}\tJoint Pos 3 : {torch.rad2deg(joint_pos)[8]}")

        projected_gravity = Custom._calculate_projected_acceleration(quat)

        # self.recieve_controller_commands()
        self.recieve_controller_commands_keyboard()
        scaled_commands = self.commands.clone().to(self.config.device) 

        episode_time = self.loop_count * self.dt
        phase = (episode_time % self.gait_period) / self.gait_period
        sine_phase = torch.sin(torch.tensor([2 * torch.pi * phase], dtype=torch.float32, device=self.config.device))
        cosine_phase = torch.cos(torch.tensor([2 * torch.pi * phase], dtype=torch.float32, device=self.config.device))
        
        current_obs = TensorDict({
            "policy": torch.cat([
                gyro * 0.5,                     # 3
                projected_gravity,              # 3
                joint_pos,                      # 12
                joint_vel * 0.05,               # 12
                self.actions,                   # 12
                scaled_commands,                # 3
                sine_phase,                     # 1
                cosine_phase,                   # 1
            ], dim=0).unsqueeze(0),                          
        }, batch_size=[1], device=self.config.device) 

        return current_obs
    
    def get_actions(self):

        if self.low_state is not None:
            quat = self.low_state.imu_state.quaternion
            gz = -(1.0 - 2.0 * (quat[1]**2 + quat[2]**2))
            if gz > -0.5:
                print(f"\n[ALERT] Fall detected (gz = {gz:.2f})! Triggering SafeShutdown.")
                self.SafeShutdown()
                return
            
        current_obs = self._get_obs_space()["policy"]
        print(f"commands: {self.commands}")
        print(f"obs tail: {current_obs[0, 44:47]}")
        with torch.no_grad():
            actions = self.policy(current_obs)
            
        self.actions = actions.squeeze()
        self.actions_scaled = self.config.initial_positions + actions.squeeze() * self.config.action_scale
        if self.activate_joint_limits:
            lower = self.config.clamped_joint_limits[:, 0]
            upper = self.config.clamped_joint_limits[:, 1]
            self.actions_scaled = torch.clamp(self.actions_scaled, lower, upper)
            
        self.actions_to_motor = self.actions_scaled[self.SIM2REAL]
        self.motor_target = self.actions_to_motor.clone()

    
    def clamp_actions(self):
        for i in range(12):
            self.actions_to_motor[i] = max(self.config.clamped_joint_limits[i][0], min(self.actions_to_motor[i], self.config.clamped_joint_limits[i][1]))
    
    def SafeShutdown(self):
        """Puts all motors into damping mode and shuts down cleanly."""
        print("\n[SafeShutdown] Engaging damping mode...")
        self.emergency_stop = True

        # Stop low-level write thread if running
        if hasattr(self, 'lowCmdWriteThreadPtr') and self.lowCmdWriteThreadPtr is not None:
            try:
                # Signal thread to stop if supported
                pass
            except Exception:
                pass

        # Send damping commands continuously for 0.5 seconds so robot settles smoothly
        t_end = time.time() + 0.5
        while time.time() < t_end:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = self.Kd  # Gentle damping
                self.low_cmd.motor_cmd[i].tau = 0.0

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)
            time.sleep(0.005)

        print("[SafeShutdown] Motors safely damped. Exit complete.")


    def MotorWrite(self):
        if self.low_state is None:
            return

        if self.emergency_stop:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)
            return
        
        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False

            print(f"Start position: {self.startPos}")

        
        # self.get_actions()

        # print(f"Actions to motor: {self.actions_to_motor}")

        
        initial_position = self.config.initial_positions[self.SIM2REAL]
        if self.not_in_stand_position:
            self.stand_percent += 1.0 / self.stand_duration
            self.stand_percent = min(self.stand_percent, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.stand_percent) * self.startPos[i] + self.stand_percent * float(initial_position[i])
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0
            if self.stand_percent == 1.0:
                self.not_in_stand_position = False
                self.activate_joint_limits = True
                self.prev_actions_to_motor = self.config.initial_positions[self.SIM2REAL].clone()
                self.loop_count = 0
        else:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = float(self.motor_target[i])
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        self.loop_count += 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def load_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["actor_state_dict"]

    actor = ActorMLP()
    actor.obs_mean.copy_(sd["obs_normalizer._mean"])
    actor.obs_std.copy_(sd["obs_normalizer._std"])

    mlp_sd = {k.replace("mlp.", ""): v for k, v in sd.items() if k.startswith("mlp.")}
    actor.mlp.load_state_dict(mlp_sd)
    actor.to(device)
    actor.eval()

    return actor

if __name__ == '__main__':

    config = WalkEnvCfg()
    ChannelFactoryInitialize(0) 
    actor = load_policy("/home/yayy/My/Codeeeeee/RL_Robotics/go2_sim2real/logs/go2_walk_seven/model_10000.pt", device=config.device)

    custom = Custom(config=config, policy=actor)
    custom.Init()
    # custom.Start()

    # try:
    #     while not custom.emergency_stop:
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     print("\n[Ctrl+C] Stopping...")
    # finally:
    #     custom.SafeShutdown()

    try:
        custom.Start()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping...")
    finally:
        custom.SafeShutdown()