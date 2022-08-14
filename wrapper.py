import gym
import numpy as np


class NormFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        frame = frame / 255.
        return frame