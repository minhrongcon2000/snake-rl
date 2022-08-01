from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines3 import DQN
import gym
import snake_gym_grid.snake_gym_grid
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import argparse

def make_snake_env():
    env = make_vec_env("snake-gym-grid-10x20-v0", 4)
    env = VecFrameStack(env, 2)
    env = VecNormalize(env, clip_obs=1.0)
    return env


parser = argparse.ArgumentParser()
parser.add_argument("--wandb_api_key", type=str)
parser.add_argument("--total_timestep", type=int)
args = parser.parse_args()

key = args.wandb_api_key

os.environ["WANDB_API_KEY"] = key

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": args.total_timestep,
    "env_name": "snake-gym-grid-10x20-v0",
}
run = wandb.init(
    project="snake_rl_sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
    name="DQN"
)

env = make_snake_env()
model = DQN("CnnPolicy", env, tensorboard_log=f"runs/{run.id}", buffer_size=100000)
model.learn(
    total_timesteps=config["total_timesteps"], 
    callback=WandbCallback(
        model_save_path=f"models/{run.id}", 
        verbose=2
    )
)
run.finish()
