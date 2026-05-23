import os
import shutil
import pickle
import torch
import genesis as gs
import rsl_rl
from rsl_rl.runners import OnPolicyRunner
import yaml
import argparse
from src.env import WalkENV
from configs.env_configs import WalkEnvCfg

def main():
    gs.init(backend=gs.gpu, logging_level="warning")


    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2_train_first")
    parser.add_argument("-r", "--resume", action="store_true", help="Whether to resume training from the last checkpoint")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint number from where to resume training.")
    args = parser.parse_args()

    exp_name = args.exp_name
    num_envs = 4096
    max_iterations = 10000

    config = WalkEnvCfg(device=gs.device, num_envs=num_envs)

    env = WalkENV(config=config, render=False, plane=True)

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