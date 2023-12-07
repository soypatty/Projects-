import gym
from gym import spaces
import numpy as np

class FastestRouteFindingRobotEnv(gym.Env):
    def __init__(self, map_size=10):
        super(FastestRouteFindingRobotEnv, self).__init__()

        self.observation_space = spaces.Discrete(map_size)
        self.action_space = spaces.Discrete(3) 
        self.robot_position = 0
        self.goal_position = map_size - 1

    def reset(self):
        self.robot_position = 0
        return self.robot_position

    def step(self, action):
        if action == 0:
            self.robot_position = min(self.robot_position + 1, self.observation_space.n - 1)
        elif action == 1:
            self.robot_position = max(self.robot_position - 1, 0)

        done = (self.robot_position == self.goal_position)
        reward = -abs(self.robot_position - self.goal_position) 

        done = done or (self.robot_position == self.observation_space.n - 1)

        return self.robot_position, reward, done, {}
