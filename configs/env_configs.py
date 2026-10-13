from dataclasses import dataclass
from .sim_config import SimCfg
from .robot_config import RobotCfg
from .task_config import TaskCfg
from .domain_randomize_config import DomainRandomizationCfg


@dataclass
class WalkEnvCfg(SimCfg, RobotCfg, TaskCfg, DomainRandomizationCfg):
    pass
