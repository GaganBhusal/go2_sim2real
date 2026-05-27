import time
import sys
import torch
import TensorDict
from src.utils.math_utils import _calculate_projected_acceleration, _quat_to_euler


from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
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

class Custom:
    def __init__(self):
        self.Kp = self.config.kp
        self.Kd = self.config.kv
        self.dt = 0.005  # 0.001~0.01

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.startPos = [0.0] * 12
        self.SIM2REAL = [0, 4, 8,
                         1, 5, 9,
                         2, 6, 10, 
                         3, 7, 11]
        self.actions_to_motor = torch.zeros((12, ), device=self.config.device, dtype=torch.float32)
        self.not_in_stand_position = True

        # thread handling
        self.lowCmdWriteThreadPtr = None

        self.crc = CRC()

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
            interval=0.002, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowStatecan):
        self.low_state = msg
        # print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]])
        # print("IMU state: ", msg.imu_state)
        # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)

    def recieve_controller_commands(self):
        pass
    
    def _get_obs_space(self):
        msg = self.low_state  

        gyro = torch.tensor(msg.imu_state.gyroscope, dtype=torch.float32)
        quat = msg.imu_state.quaternion
        euler = _quat_to_euler(quat)

        joint_pos = torch.tensor([msg.motor_state[i].q  for i in range(12)], dtype=torch.float32)
        joint_vel = torch.tensor([msg.motor_state[i].dq for i in range(12)], dtype=torch.float32)

        projected_gravity = _calculate_projected_acceleration(quat)


        scaled_commands = "we will get these from real controller will write it later!!!" 

        current_obs = TensorDict({
            "policy": torch.cat([
                gyro * 0.5,                     #
                projected_gravity,
                euler[:, :2].unsqueeze(0),          
                joint_pos,
                joint_vel * 0.05,
                self.actions,
                scaled_commands,
            ], dim=0),                          
        }, batch_size=[1], device=self.config.device)        

        return current_obs
    
    def get_actions(self):
        current_obs = self._get_obs_space()
        actions = self.policy(current_obs)

        self.actions_to_motor = actions[self.SIM2REAL]  # ← reorder here


    def LowCmdWrite(self):
        # I have to smooth this out or it might go wrong as giving such high values immediately
        if self.not_in_stand_position:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self.config.initial_position
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0
            self.not_in_stand_position = False


        else:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self.actions_to_motor[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:        
        if custom.percent_4 == 1.0: 
           time.sleep(1)
           print("Done!")
           sys.exit(-1)     
        time.sleep(1)