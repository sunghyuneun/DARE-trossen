"""
Gymnasium environment for WidowX AI pick-and-place, compatible with Stable-Baselines3.

The policy controls the arm in joint space: each action is a vector of normalized joint
velocities (6 arm joints) plus a normalized gripper opening.
"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "configs" / "wxai_pick_place.yaml"


class Trossen_Env(gym.Env):
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        robot_cfg = self.config["robot"]
        control_cfg = self.config["control"]
        task_cfg = self.config["task"]

        self.model = mujoco.MjModel.from_xml_path(
            str(PACKAGE_DIR / self.config["scene_xml_path"])
        )
        self.data = mujoco.MjData(self.model)

        # Robot indices
        self.arm_qpos_indices = np.array(
            [self.model.joint(name).qposadr[0] for name in robot_cfg["arm_joint_names"]]
        )
        self.gripper_qpos_indices = np.array(
            [self.model.joint(name).qposadr[0] for name in robot_cfg["gripper_joint_names"]]
        )
        self.arm_actuator_ids = np.array(
            [self.model.actuator(name).id for name in robot_cfg["arm_actuator_names"]]
        )
        self.gripper_actuator_id = self.model.actuator(robot_cfg["gripper_actuator_name"]).id
        self.ee_site_id = self.model.site(robot_cfg["ee_site_name"]).id

        self.home_qpos = np.array(robot_cfg["home_qpos"], dtype=np.float64)
        self.gripper_open_position = robot_cfg["gripper_open_position"]

        # Cube indices (freejoint: 7 qpos = 3 pos + 4 quat, 6 qvel)
        self.cube_body_id = self.model.body(task_cfg["cube_body_name"]).id
        cube_joint_id = self.model.body_jntadr[self.cube_body_id]
        if cube_joint_id < 0:
            raise RuntimeError("Could not find joint for cube body")
        self.cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.cube_qvel_addr = self.model.jnt_dofadr[cube_joint_id]

        self.cube_initial_position = np.array(task_cfg["cube_initial_position"])
        self.cube_initial_orientation = np.array(task_cfg["cube_initial_orientation"])
        self.target_position = np.array(task_cfg["target_position"])

        # Control
        self.frame_skip = control_cfg["frame_skip"]
        self.control_dt = self.model.opt.timestep * self.frame_skip
        self.max_joint_velocity = np.array(control_cfg["max_joint_velocity"])
        self.max_episode_steps = control_cfg["max_episode_steps"]
        self.arm_ctrl_low, self.arm_ctrl_high = self.model.actuator_ctrlrange[
            self.arm_actuator_ids
        ].T
        self.gripper_ctrl_low, self.gripper_ctrl_high = self.model.actuator_ctrlrange[
            self.gripper_actuator_id
        ]

        # Position actuator targets, integrated from joint velocity actions
        self.arm_target = self.home_qpos.copy()
        self.step_count = 0

        # Actions: 6 arm joint velocities + 1 gripper opening, all in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.arm_actuator_ids) + 1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._get_obs().shape,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        mujoco.mj_resetData(self.model, self.data)
        self._reset_robot()
        self._reset_cube(
            position=options.get("cube_position"),
            orientation=options.get("cube_orientation"),
        )
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Arm: integrate joint velocities into position targets
        self.arm_target = np.clip(
            self.arm_target + action[:6] * self.max_joint_velocity * self.control_dt,
            self.arm_ctrl_low,
            self.arm_ctrl_high,
        )
        self.data.ctrl[self.arm_actuator_ids] = self.arm_target

        # Gripper: -1 = closed, +1 = open
        self.data.ctrl[self.gripper_actuator_id] = np.interp(
            action[6], [-1.0, 1.0], [self.gripper_ctrl_low, self.gripper_ctrl_high]
        )

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.step_count += 1

        obs = self._get_obs()
        reward = 0.0  # TODO: Implement your task reward.
        terminated = False  # TODO: Detect task completion or failure.
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def get_cube_pose(self):
        """Get current cube position [x, y, z] and orientation quaternion [w, x, y, z]."""
        return (
            self.data.xpos[self.cube_body_id].copy(),
            self.data.xquat[self.cube_body_id].copy(),
        )

    def get_ee_position(self):
        """Get current end effector position [x, y, z]."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def _reset_robot(self):
        """Reset arm to home pose with the gripper open."""
        self.data.qpos[self.arm_qpos_indices] = self.home_qpos
        self.data.qpos[self.gripper_qpos_indices] = self.gripper_open_position

        self.arm_target = self.home_qpos.copy()
        self.data.ctrl[self.arm_actuator_ids] = self.arm_target
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_open_position

    def _reset_cube(self, position=None, orientation=None):
        """Reset cube to specified or initial pose."""
        position = position if position is not None else self.cube_initial_position
        orientation = orientation if orientation is not None else self.cube_initial_orientation

        self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3] = position
        self.data.qpos[self.cube_qpos_addr + 3 : self.cube_qpos_addr + 7] = orientation
        self.data.qvel[self.cube_qvel_addr : self.cube_qvel_addr + 6] = 0.0

    def _get_obs(self):
        # TODO: Select meaningful robot state variables.
        return np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
            ]
        ).astype(np.float32)

