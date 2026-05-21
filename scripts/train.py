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
    args = parser.parse_args()

    exp_name = args.exp_name
    num_envs = 4096
    max_iterations = 10000

    config = WalkEnvCfg(device=gs.device, num_envs=num_envs)

    env = WalkENV(config=config, render=False, plane=True)

    log_dir = f"logs/{exp_name}"

    # To start the training fresh again, removing the old logs.

    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)


    os.makedirs(log_dir, exist_ok=True)
    with open("configs/train_params.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    pickle.dump(train_cfg, open(f"{log_dir}/cfgs.pkl", "wb"))

    runner = OnPolicyRunner(env, train_cfg["runner"], log_dir, device=gs.device)


    # To resume the training from a checkpoint.
    
    # resume_path = "logs/go2_rsl/model_2000.pt"
    # print(f"Loading model from: {resume_path}")
    # runner.load(resume_path)

    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=False)

if __name__ == "__main__":
    main()