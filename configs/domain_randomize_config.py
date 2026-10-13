import torch
from dataclasses import dataclass

@dataclass
class DomainRandomizationCfg:
    mass_shift_range: tuple = (-1.0, 2.5)
    com_shift_range: tuple = (0.04, 0.04, 0.04)
    friction_range: tuple = (0.05, 3.0)
    damping_range: tuple = (0.02, 0.40)
    frictionloss_range: tuple = (0.02, 0.50)
    armature_range: tuple = (0.001, 0.008)
    kp_range: tuple = (20.0, 80.0)
    kv_range: tuple = (0.2, 5.0)
    torque_limit_scale_range: tuple = (0.7, 1.0)
    max_push_force: float = 150.0
    nominal_max_torque: tuple = (23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 35.55, 35.55, 35.55, 35.55)
    force_push_prob: float = 0.01
