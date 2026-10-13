"""
A wrapper to make the original environment compatible with Stable-Baselines3
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class Trossen_Env(gym.Env):
    def __init__(self, model_path):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjModel(self.model)

        # TODO: Match these.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # TODO: Map action to the robot's actual control interface.
        self.data.ctrl[:] = action

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0  # TODO: Implement your task reward.
        terminated = False  # TODO: Detect task completion or failure.
        truncated = False  # TODO: Implement episode time limits.

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # TODO: Select meaningful robot state variables.
        return np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
            ]
        ).astype(np.float32)
