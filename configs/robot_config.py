import torch
from dataclasses import dataclass, field

@dataclass
class RobotCfg:
    URDF_LOCATION: str = "/home/yayy/My/Codeeeeee/Simulators/Genesis/genesis-world/genesis/assets/urdf/go2/urdf/go2.urdf"
    BASE: str = "base"
    
    total_actions: int = 12
    action_scale: float = 0.25
    
    kp: float = 40.0 
    kv: float = 1.0 

    device = "cpu"

    HIPS: tuple = torch.tensor([1, 2, 3, 4], device = device)
    THIGHS: tuple = torch.tensor([5, 6, 7, 8], device = device)
    FOOTS: tuple = torch.tensor([9, 10, 11, 12], device = device)
    FOOT_LINK_NAMES: tuple = ('FL_calf', 'FR_calf', 'RL_calf', 'RR_calf')
    
    
    initial_positions = torch.deg2rad(
        torch.tensor([
            0, 0, 0, 0, 50, 50, 55, 55, -90, -90, -90, -90
        ], dtype=torch.float, device=device)
    )

    # actual_joint_limits = torch.deg2rad(
    #     torch.tensor([
    #         (-48, 48), (-48, 48), (-48, 48), (-48, 48), 
    #         (-90, 200), (-90, 200), (-30, 260), (-30, 260), 
    #         (-156, -48), (-156, -48), (-156, -48), (-156, -48)
    #     ], dtype=torch.float, device=device)
    # )

    clamped_joint_limits = torch.deg2rad(
        torch.tensor([
            (-20, 20), (-20, 20), (-20, 20), (-20, 20), 
            (-90, 200), (-90, 200), 
            (-30, 260), (-30, 260), 
            (-156, -48), (-156, -48), (-156, -48), (-156, -48)
        ], dtype=torch.float, device=device)
    )


    # joint_limits = torch.deg2rad(
    #     torch.tensor([
    #         (-40, 40), (-40, 40), (-40, 40), (-40, 40),
    #         (-180, 70), (-180, 70),
    #         (-240, 20), (-240, 20),
    #         (-140, -40), (-140, -40), (-140, -40), (-140, -40)
    #     ], dtype=torch.float, device=device)
    # )

    LEG_OFFSET = torch.tensor([0.178, 0.178, 0.194, 0.194], dtype=torch.float , device=device)
