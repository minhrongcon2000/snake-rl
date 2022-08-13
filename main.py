from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines3 import PPO
import gym
from snake_gym_grid.snake_gym_grid.envs.snake_gym_grid import SnakeGymGrid
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import argparse


def make_snake_env():
    def wrap_single_env(env):
        env = WarpFrame(env)
        env = MaxAndSkipEnv(env, 2)
        return env
    env = make_vec_env(SnakeGymGrid, 4, wrapper_class=wrap_single_env)
    env = VecFrameStack(env, 2)
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
    name="PPO"
)

env = make_snake_env()
model = PPO("CnnPolicy", env, tensorboard_log=f"runs/{run.id}", buffer_size=100000)
model.learn(
    total_timesteps=config["total_timesteps"], 
    callback=WandbCallback(
        model_save_path=f"models/{run.id}", 
        verbose=2
    )
)
run.finish()
