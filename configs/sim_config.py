import torch
from dataclasses import dataclass

@dataclass
class SimCfg:
    device: str = "cpu"
    num_envs: int = 4096
    max_episode_length: int = 1000
    
    sim_substeps: int = 10
    enable_self_collision: bool = True
    max_collision_pairs: int = 200
    dt: float = 0.02
    global_gravity: tuple = (0.0, 0.0, -9.81)
