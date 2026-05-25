import argparse
import os
import pickle
import time
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from src.env import WalkENV
import pygame  # Used for keyboard event handling
from configs.env_configs import WalkEnvCfg  



def get_keyboard_input():
    """
    Reads WASD inputs from keyboard.
    W/S: Forward/Backward (+1.0 to -1.0)
    A/D: Turn Left/Right (+1.0 to -1.0)
    """
    pygame.event.pump() 
    keys = pygame.key.get_pressed()
    
    vx = 0.0
    wz = 0.0

    # W/S for linear velocity (X-axis)
    if keys[pygame.K_w]:
        vx = 1.0
    elif keys[pygame.K_s]:
        vx = -1.0

    # A/D for angular velocity (Z-axis / Turning)
    # Pygame typical convention: D turns right (negative or positive depending on your env setup)
    # Usually, positive wz turns left, negative turns right.
    if keys[pygame.K_a]:
        wz = 1.0
    elif keys[pygame.K_d]:
        wz = -1.0

    return vx, wz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_train_first")
    parser.add_argument("--ckpt", type=int, default=380)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Initialize pygame and open a small window to capture focus for keyboard inputs
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Keyboard Control Focus")
    print("Keep the Pygame window focused to send WASD commands!")

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
    MAX_WZ = 1.2  

    with torch.no_grad():
        while True:
            # Handle quit event to close gracefully
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get WASD inputs
            kbd_vx, kbd_wz = get_keyboard_input()
            
            target_vx = kbd_vx * MAX_VX
            target_wz = kbd_wz * MAX_WZ

            # Interpolate for smoother transitions
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