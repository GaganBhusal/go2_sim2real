import torch
from dataclasses import dataclass

@dataclass
class WalkEnvCfg:
    device: str = "cuda"
    num_envs: int = 4096
    max_episode_length: int = 1000

    max_episode_length: int = 1000

    # For commands
    vx_range: tuple = (-1.0, 1.0)
    vy_range: tuple = (-1.0, 1.0)
    wz_range: tuple = (-0.6, 0.6)
    height_range: float = (0.22, 0.32)
    pitch_range: tuple = (-0.15, 0.15)

    action_scale: float = 0.5
    dt: float = 0.02
    kp: float = 40.0
    kv: float = 1.0

    sigma_vx: float = 0.25
    sigma_wz: float = 0.10
    global_gravity: torch.Tensor = torch.tensor([0, 0, -1], device=device)

    initial_positions = torch.deg2rad(
        torch.tensor([
            0, 0, 0, 0, 50, 50, 55, 55, -90, -90, -90, -90
        ]).to(device)
    )
    # The hip joints can be set between -90 to -125 for different height based movement and according to requirements.

    actual_joint_limits = torch.deg2rad(torch.tensor([(-48, 48), (-48, 48),(-48, 48), (-48, 48), 
                                                      (-90, 200), (-90, 200), (-30, 260), (-30, 260), 
                                                      (-156, -48),(-156, -48),(-156, -48), (-156, -48)
        ])
    ).to(device)

    # The joint limits defined in the URDF

    URDF_specified_joint_limits = torch.deg2rad(
        torch.tensor([
            (-10, 10),              # actual urdf specified is (-60 to 60) but limitting hip movement further to prevent excessive joint movement. 
            (-10, 10),              # actual urdf specified is (-60 to 60) but limitting hip movement further to prevent excessive joint movement.
            (-10, 10),              # actual urdf specified is (-60 to 60) but limitting hip movement further to prevent excessive joint movement. 
            (-10, 10),              # actual urdf specified is (-60 to 60) but limitting hip movement further to prevent excessive joint movement. 

            (-90, 200), 
            (-90, 200), 

            (-30, 260), 
            (-30, 260), 

            (-156, -48), 
            (-156, -48), 
            (-156, -48), 
            (-156, -48)
        ])
    ).to(device)

    # Joint limits I used while training
    joint_limits = torch.deg2rad(
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
    ).to(device)
