import tianshou as ts
import torch
import gym
import snake_gym_grid
from model import MLPNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chkpt_dir", type=str, required=True, help="model file to test")
parser.add_argument("--device", type=str, default="cpu", help="whether to use GPU for testing", choices=["cuda", "cpu"])
args = parser.parse_args()

env = gym.make("snake-gym-grid-10x20-1d-v0")
feature_shape = 8
num_actions = 4

model = MLPNet(feature_shape, num_actions, device=args.device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(
    model=model,
    optim=optim,
)

policy.load_state_dict(torch.load(args.chkpt_dir, map_location=args.device), False)
policy.eval()
policy.set_eps(0.05)
test_collector = ts.data.Collector(policy, env, exploration_noise=False)
results = test_collector.collect(n_episode=1, render=1 / 35)
print(results)
