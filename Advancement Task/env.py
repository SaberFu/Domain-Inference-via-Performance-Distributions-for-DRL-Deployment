# Training Env of Advancement Task 

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import hexa
import time


class LegController(gym.Env):
    def __init__(self):
        super(LegController, self).__init__()

        self._ctrl_cost_weight = 600
        self._angle_weight = 9000
        self._distance_weight = 90000

        self.noise = False
        self.std_noise = 0.16
        self.motor_model = False
        self.friction = False

        self.robo = hexa.Hexa()
        self.reward = 0.0
        self.action = np.zeros(18)
        self.obs = np.zeros(20)
        self.init_angle = 0
        self.angle = 0
        self.pos = np.zeros(3)

        high = np.ones(18) * math.pi / 6
        high = high.astype(np.float32)
        low = -high

        high2 = np.ones(20) * math.pi / 6
        high2 = high2.astype(np.float32)
        high2[-1] = 90
        high2[-2] = 90
        low2 = -high2

        # Define Observation and Action Spaces
        self.action_space = spaces.Box(low, high)
        self.observation_space = spaces.Box(low2, high2)

        # Reset state and time
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.reward = 0
        self.angle = 0
        self.action = np.zeros(18)
        self.obs = np.zeros(20)
        self.robo.reset()

        if self.friction is True:
            friction = np.random.rand() * 0.5 + 0.5
            self.robo.model.geom_friction[0][0] = friction

        if self.motor_model is True:
            kp = np.random.rand()*2+0.1
            kd = kp/10 + np.random.rand()*0.1
            self.robo.pd = [hexa.PDController(kp=kp, kd=kd) for _ in range(18)]

        # block random init
        self.robo.model.body_pos[-1][0] = np.random.uniform(-0.5, 0.5)
        self.robo.model.body_pos[-2][0] = np.random.uniform(-0.5, 0.5)

        self.robo.model.body_pos[-1][1] = np.random.uniform(0.4, 0.6)
        self.robo.model.body_pos[-2][1] = np.random.uniform(1.5, 3.5)

        # # terrain angle
        # angle = np.random.uniform(-math.pi / 12, math.pi / 12)
        # while abs(angle) < math.radians(5):
        #     angle = np.random.uniform(-math.pi / 12, math.pi / 12)
        # self.init_angle = angle
        # angle2 = np.random.uniform(0, math.pi)
        # self.robo.model.geom_quat[0] = (
        #     np.array([math.cos(angle / 2),
        #               math.sin(angle / 2) * math.cos(angle2),
        #               math.sin(angle / 2) * math.sin(angle2),
        #               0]))

        for _ in range(400):
            self.robo.step(np.zeros(18))

        self.obs = self._get_obs()
        self.angle = self.robo.get_rotate_angle
        self.pos = self.robo.get_pos.copy()

        return self._get_obs(), {}

    def _get_obs(self):
        euler = self.robo.get_euler
        q = self.robo.get_q

        state = []
        for x in q:
            state.append(x)
        for i in range(2):
            state.append(euler[i])

        state = np.array(state, dtype=np.float32)

        if self.noise is True:
            state = self.add_noise(state)

        return np.array(state, dtype=np.float32)

    def _get_rew(self):
        reward_angle = self.angle - self.robo.get_rotate_angle
        rewards = reward_angle * self._angle_weight
        reward_distance = self.robo.get_pos[1] - self.pos[1]
        rewards += reward_distance * self._distance_weight
        costs = self.control_cost()
        reward = rewards - costs
        reward_info = {
            "reward_ctrl": -costs,
            "reward_track": rewards,
        }
        return reward, reward_info

    def control_cost(self):
        return self._ctrl_cost_weight * float(np.linalg.norm(self.action - self.obs[:18]))

    def step(self, action, viewer=None):
        self.action = action

        for _ in range(100):
            self.robo.step(action)
            if viewer is not None:
                viewer.sync()
                time.sleep(0.02)

        reward, reward_info = self._get_rew()

        obs = self._get_obs()
        self.obs = obs
        self.angle = self.robo.get_rotate_angle
        self.pos = self.robo.get_pos.copy()
        self.reward += reward

        success_goal = False
        if self.pos[1] > 4:
            success_goal = True
            print(1, self.reward)

        stop = False
        if self.robo.data.time > 60:
            stop = True

        return obs, reward, stop, success_goal, reward_info

    def add_noise(self, obs):
        noise = np.random.normal(0, self.std_noise, 20)
        noise[-1] = noise[-1] / math.pi * 6 * 90
        noise[-2] = noise[-2] / math.pi * 6 * 90
        obs += noise
        return obs
