import argparse
import os
import pickle
import time
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from src.env import WalkENV
import pygame  # Added for controller support
from configs.env_configs import WalkEnvCfg  
from src.env import AdvWalkENV


def get_joystick_input(joystick, deadzone=0.1):

    pygame.event.pump() 
    raw_vx = -joystick.get_axis(1) 
    raw_wz = joystick.get_axis(3) 

    vx = raw_vx if abs(raw_vx) > deadzone else 0.0
    wz = raw_wz if abs(raw_wz) > deadzone else 0.0

    return vx, wz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_train_first")
    parser.add_argument("--ckpt", type=int, default=380)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("No controller detected!!!.")
        return
        
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Controller connected: {joystick.get_name()}")

    gs.init(backend=gs.gpu if args.device == "cuda" else gs.cpu, logging_level="warning")
    log_dir = f"logs/{args.exp_name}"

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        train_cfg = pickle.load(f)

    config = WalkEnvCfg(device=gs.device, num_envs=1)
    env = WalkENV(
        config=config,
        render=True,
        t_x=8,
        t_y=10,
        number_of_lanes=1,
        number_of_rows=5,
        plane=True,
        domain_randomize=False
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

    MAX_VX = 1.0  
    MAX_WZ = 1.2  #

    with torch.no_grad():
        while True:
            stick_vx, stick_wz = get_joystick_input(joystick)
            
            target_vx = stick_vx * MAX_VX
            target_wz = stick_wz * MAX_WZ

            env.custom_commands[:, 0] += (target_vx - env.custom_commands[:, 0]) * 0.1
            env.custom_commands[:, 1] += (target_wz - env.custom_commands[:, 1]) * 0.1

            actions = policy(obs)
            step_out = env.step(actions)
            
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            else:
                obs = step_out[0]
                
            obs = obs.to(args.device)

            time.sleep(config.dt)

if __name__ == "__main__":
    main()