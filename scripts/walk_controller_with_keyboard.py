import torch
import argparse
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import pickle
import time
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from src.env.basic_walk_without_explicit_lin_vel import WalkWithoutVx
from src.env import QuadrupedWalk
import pygame  
from configs.env_configs import WalkEnvCfg
import keyboard
import sys
# pyrefly: ignore [missing-import]
from genesis.utils.geom import transform_by_quat
import sys
# Point Python to the parent directory (ran_sim2real)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def is_key_pressed(*keys):
    for k in keys:
        try:
            if keyboard.is_pressed(k):
                return True
        except Exception:
            pass
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_train_first")
    parser.add_argument("--ckpt", type=int, default=380)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # --- Initialize Pygame and Controller ---
    pygame.init()


    gs.init(backend=gs.gpu if args.device == "cuda" else gs.cpu, logging_level="warning")
    log_dir = f"logs/{args.exp_name}"

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        train_cfg = pickle.load(f)

    config = WalkEnvCfg(device=args.device, num_envs=1, height_range=0.31, max_episode_length=100000000)

    env = QuadrupedWalk(
        config=config, 
        render=True,
        device=args.device,
        plane=True, domain_randomize=False, terrain=False
    )

    runner = OnPolicyRunner(env, train_cfg["runner"], log_dir, device=args.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    print(f"Loading checkpoint from {resume_path}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=args.device)

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out
    obs = obs.to(args.device)
# Max speed scaling factors

    
    MAX_VX = 1.0     
    MAX_VY = 0.5     
    MAX_WZ = 1.2     
    
    # NEW Height parameters
    DEFAULT_HEIGHT = 0.4
    MAX_HEIGHT = 0.32
    MIN_HEIGHT = 0.22
    HEIGHT_STEP = 0.002 # How fast the dog crouches/stands per frame

    # # Track the current physical target height outside the loop
    # target_height = DEFAULT_HEIGHT

    # # Initialize the commands buffer to 0 (and height to default)
    # env.custom_commands[:, :3] = 0.0
    # env.custom_commands[:, 3] = DEFAULT_HEIGHT

    with torch.no_grad():
        while True:

            raw_vx = 0.0
            raw_vy = 0.0
            raw_wz = 0.0
            
            # 1. Read keyboard inputs
            if is_key_pressed('up'): raw_vx += 1.0
            if is_key_pressed('down'): raw_vx -= 1.0
            
            if is_key_pressed('left'): raw_vy += 1.0
            if is_key_pressed('right'): raw_vy -= 1.0
            
            if is_key_pressed('1', 'numpad 1', 'kp_1', 'end'): raw_wz += 1.0
            if is_key_pressed('3', 'numpad 3', 'kp_3', 'page down'): raw_wz -= 1.0

            # 2. Scale stick inputs to actual environment command ranges
            target_vx = raw_vx * MAX_VX
            target_vy = raw_vy * MAX_VY
            target_wz = raw_wz * MAX_WZ
            
            # 3. Handle D-pad height adjustment
            # if hat_y == 1:    # D-pad UP
            #     target_height += HEIGHT_STEP
            # elif hat_y == -1: # D-pad DOWN
            #     target_height -= HEIGHT_STEP

            # # Clamp the target height so it cannot exceed 0.22 or 0.32
            # target_height = max(MIN_HEIGHT, min(target_height, MAX_HEIGHT))

            # 4. Apply a low-pass filter to smooth the commands (prevents jerking)
            env.custom_commands[:, 0] += (target_vx - env.custom_commands[:, 0]) * 0.1
            env.custom_commands[:, 1] += (target_vy - env.custom_commands[:, 1]) * 0.1
            env.custom_commands[:, 2] += (target_wz - env.custom_commands[:, 2]) * 0.1
            # env.custom_commands[:, 3] += (target_height - env.custom_commands[:, 3]) * 0.1

            # print(env.custom_commands)
            # 5. Step the environment
            actions = policy(obs)
            # print(f"FL : {env.feet_pos[:, 0, 2].item()}\t  FR : {env.feet_pos[:, 1, 2].item()} \nRL : {env.feet_pos[:, 2, 2].item()}\t  RR : {env.feet_pos[:, 3, 2].item()}\n\n")
            # print(env.feet_pos[:, :, 2] - config.LEG_OFFSET)

            calf_pos = torch.stack([env.robot.get_link(name).get_pos() for name in config.FOOT_LINK_NAMES], dim=1)
            calf_quat = torch.stack([env.robot.get_link(name).get_quat() for name in config.FOOT_LINK_NAMES], dim=1)

            local_foot_offset = torch.tensor([0.0, 0.0, -0.235], device=args.device)
            feet_pos = calf_pos + transform_by_quat(
                local_foot_offset.unsqueeze(0).unsqueeze(0).expand_as(calf_pos), 
                calf_quat
            )
            # print(feet_pos[:, :, 2], env.feet_pos[:, :, 2])

            # print(env.feet_air_time)
            # print(actions.shape)  
            # debug_values = torch.rad2deg(actions.squeeze(0))

            # # self.joint_pos_sim_deg = self.actions_to_motor
            # print(f"Actions to motor: {debug_values[0]}\t{debug_values[4]}\t{debug_values[8]}\n"
            #       f"{debug_values[1]}\t{debug_values[5]}\t{debug_values[9]}\n"
            #       f"{debug_values[2]}\t{debug_values[6]}\t{debug_values[10]}\n"
            #       f"{debug_values[3]}\t{debug_values[7]}\t{debug_values[11]}")
            step_out = env.step(actions)
            
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            else:
                obs = step_out[0]
                
            obs = obs.to(args.device)
            # print(env.base_z)

            time.sleep(config.dt)

if __name__ == "__main__":
    main()