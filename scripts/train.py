import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import shutil
import pickle
import torch
import genesis as gs
import rsl_rl
from rsl_rl.runners import OnPolicyRunner
import yaml
import argparse
from src.env import WalkWithoutVx, QuadrupedWalk
from configs.env_configs import WalkEnvCfg

def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_train_first")
    parser.add_argument("-d", "--domain", action="store_true", help="Whether to enable domain randomization")
    parser.add_argument("-r", "--resume", action="store_true", help="Whether to resume training from the last checkpoint")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint number from where to resume training.")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--render", action="store_true", help="Whether to show the simulator window")
    parser.add_argument("--max_iter", type=int, default=10000, help="Number of total training iterations")
    args = parser.parse_args()

    exp_name = args.exp_name
    num_envs = args.num_envs
    render = args.render
    max_iterations = args.max_iter

    config = WalkEnvCfg(device=gs.device, num_envs=num_envs)

    env = QuadrupedWalk(config=config, render=args.render, plane=True, domain_randomize=args.domain)

    log_dir = f"logs/{exp_name}"

    os.makedirs(log_dir, exist_ok=True)
    with open("configs/train_params.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    pickle.dump(train_cfg, open(f"{log_dir}/cfgs.pkl", "wb"))

    runner = OnPolicyRunner(env, train_cfg["runner"], log_dir, device=gs.device)
    # To resume the training from a checkpoint.

    if args.resume:
        if args.ckpt:
            resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")    
            print(f"Resuming training from checkpoint: {resume_path}")
            runner.load(resume_path)
        else:
            print(f"Try Again!!\nResume path provided but not checkpoint number")
    else:
        print(f"Starting training from scratch. No resume path provided.")
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()