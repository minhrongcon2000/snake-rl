import os
import random

import numpy as np
import wandb
from env_wrapper import wrap_deepmind
from model import DQN
import tianshou as ts
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

def make_atari_env(env_id, frame_stack=4):
    return wrap_deepmind(env_id=env_id, frame_stack=frame_stack)


def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def test_dqn(env_id,
             action_space,
             wandb_api_key,
             frame_stack=4,
             train_num=10,
             test_num=1,
             lr=0.0001,
             gamma=0.99,
             target_update_freq=500,
             epochs=1000,
             step_per_epoch=100000,
             step_per_collect=10,
             update_per_step=0.1,
             batch_size=64,
             buffer_size=100000,
             eps_train_start=1,
             eps_train_min=0.05,
             eps_test=0.005,
             n_step=1):
    os.environ["WANDB_API_KEY"] = wandb_api_key
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_envs = ts.env.ShmemVectorEnv(
        [lambda: make_atari_env(env_id) for _ in range(train_num)])
    test_envs = ts.env.ShmemVectorEnv(
        [lambda: make_atari_env(env_id) for _ in range(test_num)])

    net = DQN(4, 84, 84, action_space, device=device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optimizer,
        discount_factor=gamma,
        target_update_freq=target_update_freq,
        estimation_step=n_step
    )

    buffer = ts.data.VectorReplayBuffer(
        total_size=buffer_size,
        buffer_num=train_num,
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=frame_stack
    )
    
    train_collector = ts.data.Collector(
        policy=policy,
        env=train_envs,
        buffer=buffer,
        exploration_noise=True
    )
    test_collector = ts.data.Collector(
        policy=policy,
        env=test_envs,
        buffer=buffer,
        exploration_noise=True
    )
    wandb.init(settings=wandb.Settings(start_method='thread'))
    logger = ts.utils.WandbLogger(save_interval=1,
                                  name="snake-dqn",
                                  project="snake-rl",
                                  name="dqn")
    logger.load(SummaryWriter("logs"))
    
    
    def save_fn(policy):
        torch.save(policy.state_dict(), "policy.pth")
        
    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = eps_train_start - env_step / 1e6 * (eps_train_start - eps_train_min)
        else:
            eps = eps_train_min
        
        policy.set_eps(eps)
        
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
            
    def test_fn(epoch, env_step):
        policy.set_eps(eps_test)
        
    def stop_fn(mean_reward):
        return mean_reward >= 20 * 10 - 2
    
    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=batch_size * train_num)
    # trainer
    result = ts.trainer.offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epochs,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        update_per_step=update_per_step,
        test_in_train=False,
    )
    
    return result

if __name__ == "__main__":
    set_global_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_api_key", type=str)
    args = parser.parse_args()
    wandb_api_key = args.wandb_api_key
    test_dqn("snake-gym-10x20-v0", 4, wandb_api_key)