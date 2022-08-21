import os
import tianshou as ts
import gym
import snake_gym_grid
import torch
import wandb

from model import MLPNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_train_env", 
    type=int,
    default=5,
    help="Number of parallel training env"
)
parser.add_argument(
    "--num_test_env", 
    type=int,
    default=10,
    help="Number of parallel test env"
)
parser.add_argument(
    "--pre_collect",
    type=int,
    default=500,
    help="Number of steps to collect before training",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=100,
    help="number of training epochs"
)

parser.add_argument(
    "--step_per_collect",
    type=int,
    default=100,
    help="number of step collected to train the agent per epochs"
)

parser.add_argument(
    "--episode_per_test",
    type=int,
    default=100,
    help="number of episodes per test"
)
parser.add_argument(
    "--max_eps",
    type=float,
    default=1.0,
    help="initial exploration rate"
)

parser.add_argument(
    "--min_eps",
    type=float,
    default=0.05,
    help="minimum exploration rate"
)

parser.add_argument(
    "--exp_portion",
    type=float,
    default=0.5,
    help="spend exp_portion on exploration and remaining on exploitation"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="batch size for updating"
)

parser.add_argument(
    "--wandb_api_key",
    type=str,
    help="Key for wandb"
)

parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "cuda"],
    help="whether to use cuda or cpu"
)

args = parser.parse_args()

feature_shape = 8 # specifically for snake 1D, cannot be changed
num_action = 4 # specifically for snake 1D, cannot be changed

num_train_env = args.num_train_env
num_test_env = args.num_test_env
num_epochs = args.num_epochs
pre_collect = args.pre_collect
step_per_epochs = args.step_per_collect
episode_per_test = args.episode_per_test

if args.wandb_api_key is not None:
    os.environ["WANDB_API_KEY"] = args.wandb_api_key

# for exploration step
max_eps = args.max_eps
min_eps = args.min_eps
max_time_steps = int(num_epochs * args.exp_portion)
batch_size = args.batch_size

def linear_exploration(eps, max_eps, min_eps, max_time_steps):
    return max(eps - (max_eps - min_eps) / max_time_steps, min_eps)

if __name__ == "__main__":
    # Vec env construction for both train and test
    train_envs = ts.env.ShmemVectorEnv(
        [lambda: gym.make("snake-gym-grid-10x20-1d-v0") for _ in range(num_train_env)])

    test_envs = ts.env.ShmemVectorEnv(
        [lambda: gym.make("snake-gym-grid-10x20-1d-v0") for _ in range(num_test_env)])

    # model construction
    model = MLPNet(feature_shape, num_action, device=args.device).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Double DQN because vanilla is kinda sht!
    policy = ts.policy.DQNPolicy(
        model=model,
        optim=optim,
        estimation_step=1,
        target_update_freq=320,
        is_double=True
    )

    # Data collector
    train_collector = ts.data.Collector(
        policy=policy,
        env=train_envs,
        buffer=ts.data.VectorReplayBuffer(100000, 10),
        exploration_noise=True
    )

    test_collector = ts.data.Collector(
        policy=policy,
        env=test_envs,
        exploration_noise=False
    )

    # main training loops
    train_collector.collect(
        n_step=pre_collect,
        random=True
    )
    current_eps = max_eps
    policy.set_eps(current_eps)
    print("Start training...")
    
    if args.wandb_api_key is not None:
        wandb.init(project="tianshou_snake", name="DQN")
    
    for _ in range(num_epochs):
        collect_result = train_collector.collect(n_step=step_per_epochs)
        losses = policy.update(batch_size, train_collector.buffer)
        
        # evaluation
        result = test_collector.collect(n_episode=episode_per_test)
        
        if args.wandb_api_key is not None:
            wandb.log({
                "mean_reward": result["rew"],
                "reward_std": result["rew_std"],
                "eps": current_eps,
            })
        else:
            print(result)
        current_eps = linear_exploration(current_eps, 
                                        max_eps=max_eps, 
                                        min_eps=min_eps, 
                                        max_time_steps=max_time_steps)
        policy.set_eps(current_eps)
