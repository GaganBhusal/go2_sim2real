
import torch

class DomainRandomization:
    def __init__(self, robot, n_envs, joints_local_idx, device="cuda"):
        self.robot = robot
        self.n_envs = n_envs
        self.joints_local_idx = joints_local_idx
        self.device = device
        
        self.base_link_idx = self.robot.get_link("base").idx_local
        self.base_link_global_idx = self.robot.get_link("base").idx
        self.max_delay_steps = 3  # Max steps for action latency
        self.action_buffer = torch.zeros((n_envs, self.max_delay_steps, len(joints_local_idx)), device=device)
        self.env_delay_steps = torch.randint(0, self.max_delay_steps, (n_envs,), device=device)

    def randomize(self, env_idx=None):
        # 1. Safely handle env_idx
        if env_idx is None:
            env_idx = torch.arange(self.n_envs, device=self.device)
        
        total_envs = len(env_idx)

        # Base mass shift
        base_mass_shift = (torch.rand((total_envs, 1), device=self.device) * 4.0) - 1.0
        self.robot.set_mass_shift(
            mass_shift=base_mass_shift,
            links_idx_local=self.base_link_idx,
            envs_idx=env_idx
        )

        # Base COM shift
        com_shift = (torch.rand((total_envs, 1, 3), device=self.device) - 0.5) * 0.1
        com_shift[..., 2] *= 0.4 
        self.robot.set_COM_shift(
            com_shift=com_shift,
            links_idx_local=self.base_link_idx,
            envs_idx=env_idx
        )

        # Friction
        self.robot.set_friction_ratio(
            friction_ratio=0.5 + torch.rand((total_envs, self.robot.n_links), device=self.device),
            links_idx_local=range(self.robot.n_links),
            envs_idx=env_idx
        )

        # Kp and Kv
        kp_values = torch.randint(low=30, high=60, size=(total_envs, 12), device=self.device).float()
        kv_values = torch.rand((total_envs, 12), device=self.device) * 1.5 + 0.5
        
        self.robot.set_dofs_kp(kp_values, self.joints_local_idx, envs_idx=env_idx)
        self.robot.set_dofs_kv(kv_values, self.joints_local_idx, envs_idx=env_idx)

    # Random force push applied to the base link
    def _random_force_push(self, env_idx=None):
        if env_idx is None:
            env_idx = torch.arange(self.n_envs, device=self.device)
        total_envs = len(env_idx)
        
        forces = torch.zeros((total_envs, self.base_link_global_idx, 3), device=self.device)
        forces[:, self.base_link_idx, 0] = (torch.rand(total_envs, device=self.device) - 0.5) * 2 * 150.0
        forces[:, self.base_link_idx, 1] = (torch.rand(total_envs, device=self.device) - 0.5) * 2 * 80.0
        
        self.robot.scene.rigid_solver.apply_links_external_force(
            force=forces,
            links_idx=self.base_link_global_idx,
            envs_idx=env_idx
        )

    # Add action latency by maintaining a buffer of past actions and returning the action from a few steps ago
    def _add_action_latency(self, actions, latency_steps=2):
        if latency_steps <= 0:
            return actions
        
        self.action_buffer = torch.roll(self.action_buffer, shifts=-1, dims=1)
        self.action_buffer[:, -1] = actions
        
        return self.action_buffer[torch.arange(self.n_envs, device=self.device), self.env_delay_steps, :]
