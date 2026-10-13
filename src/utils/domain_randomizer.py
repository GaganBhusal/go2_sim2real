import torch

class DomainRandomization:
    def __init__(self, robot, joints_local_idx, config, device="cuda"):
        self.robot = robot
        self.config = config
        self.n_envs = self.config.num_envs
        self.joints_local_idx = joints_local_idx
        self.device = device
        self.config = config
        
        self.base_link_idx = self.robot.get_link("base").idx_local
        self.base_link_global_idx = self.robot.get_link("base").idx
        self.max_delay_steps = 3  # Max steps for action latency
        self.action_buffer = torch.zeros((self.n_envs, self.max_delay_steps, len(joints_local_idx)), device=device)
        self.env_delay_steps = torch.randint(0, self.max_delay_steps, (self.n_envs,), device=device)

        self.n_dofs = len(self.joints_local_idx)

    def randomize(self, env_idx=None):
        if env_idx is None:
            env_idx = torch.arange(self.n_envs, device=self.device)
        
        total_envs = len(env_idx)

        # Resample delay steps for reset environments
        self.env_delay_steps[env_idx] = torch.randint(0, self.max_delay_steps, (total_envs,), device=self.device)

        # Base mass shift
        base_mass_shift = torch.rand((total_envs, 1), device=self.device) * (self.config.mass_shift_range[1] - self.config.mass_shift_range[0]) + self.config.mass_shift_range[0]
        self.robot.set_mass_shift(
            mass_shift=base_mass_shift,
            links_idx_local=self.base_link_idx,
            envs_idx=env_idx
        )

        # Base COM shift
        com_shift = (torch.rand((total_envs, 1, 3), device=self.device) - 0.5) * 2  
        com_shift[..., 0] *= self.config.com_shift_range[0]
        com_shift[..., 1] *= self.config.com_shift_range[1]
        com_shift[..., 2] *= self.config.com_shift_range[2] 
        self.robot.set_COM_shift(
            com_shift=com_shift,
            links_idx_local=self.base_link_idx,
            envs_idx=env_idx
        )

        # Friction
        friction_ratio = (torch.rand((total_envs, self.robot.n_links), device=self.device) * (self.config.friction_range[1] - self.config.friction_range[0]) + self.config.friction_range[0])
        self.robot.set_friction_ratio(
            friction_ratio=friction_ratio,
            links_idx_local=range(self.robot.n_links),
            envs_idx=env_idx
        )

        # Kp and Kv (Batched per-environment for maximum diversity)
        kp_values = torch.rand((self.n_dofs,), device=self.device) * (self.config.kp_range[1] - self.config.kp_range[0]) + self.config.kp_range[0]
        kv_values = torch.rand((self.n_dofs, ), device=self.device) * (self.config.kv_range[1] - self.config.kv_range[0]) + self.config.kv_range[0]
        
        self.robot.set_dofs_kp(kp_values, self.joints_local_idx, envs_idx=env_idx)
        self.robot.set_dofs_kv(kv_values, self.joints_local_idx, envs_idx=env_idx)

        damping = (torch.rand((total_envs, self.n_dofs), device=self.device) * (self.config.damping_range[1] - self.config.damping_range[0]) + self.config.damping_range[0])
        self.robot.set_dofs_damping(damping, dofs_idx_local=self.joints_local_idx, envs_idx=env_idx)

        frictionloss = (torch.rand((total_envs, self.n_dofs), device=self.device) * (self.config.frictionloss_range[1] - self.config.frictionloss_range[0]) + self.config.frictionloss_range[0])
        self.robot.set_dofs_frictionloss(frictionloss, dofs_idx_local=self.joints_local_idx, envs_idx=env_idx)

        # armature = (torch.rand((total_envs, self.n_dofs), device=self.device) * (self.config.armature_range[1] - self.config.armature_range[0]) + self.config.armature_range[0])
        # self.robot.set_dofs_armature(armature, dofs_idx_local=self.joints_local_idx, envs_idx=env_idx)

        self.max_torque_given_in_URDF = torch.tensor(self.config.nominal_max_torque, device=self.device)
        max_torque = self.max_torque_given_in_URDF * (torch.rand((total_envs, self.n_dofs), device=self.device) * (self.config.torque_limit_scale_range[1] - self.config.torque_limit_scale_range[0]) + self.config.torque_limit_scale_range[0])
        self.robot.set_dofs_force_range(-max_torque, max_torque, dofs_idx_local=self.joints_local_idx, envs_idx=env_idx)
# # 
        
    # Random force push applied to the base link
    def _random_force_push(self):
        all_envs = torch.arange(self.n_envs, device=self.device)
        env_idx = all_envs[torch.rand(self.n_envs, device=self.device) < self.config.force_push_prob]
        
        # Return early if no environment is selected for force push
        if len(env_idx) == 0:
            return
            
        total_envs = len(env_idx)
        
        forces = torch.zeros((total_envs, self.base_link_global_idx, 3), device=self.device)
        forces[:, self.base_link_idx, 0] = (torch.rand(total_envs, device=self.device) - 0.5) * 2 * self.config.max_push_force
        forces[:, self.base_link_idx, 1] = (torch.rand(total_envs, device=self.device) - 0.5) * 2 * self.config.max_push_force
        
        self.robot.scene.rigid_solver.apply_links_external_force(
            force=forces,
            links_idx=torch.tensor([self.base_link_global_idx], dtype=torch.long, device=self.device),
            envs_idx=env_idx
        )

    # Add action latency by maintaining a buffer of past actions
    def _add_action_latency(self, actions):
        self.action_buffer = torch.roll(self.action_buffer, shifts=-1, dims=1)
        self.action_buffer[:, -1] = actions

        return self.action_buffer[torch.arange(self.n_envs, device=self.device), self.env_delay_steps, :]

