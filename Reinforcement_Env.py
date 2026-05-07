import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import HexapodSim
import time
import yaml


def euler_to_quaternion(roll, pitch, yaw=0.0, degrees=True):
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy + sr * sp * cy

    return np.array([w, x, y, z])


class HexapodEnv(gym.Env):
    """
    Integrated Gymnasium environment for Hexapod tasks.
    Supports:
    - Task types: 'forward' and 'balance'.
    - Deterministic mode via reset(options=...).
    - Domain Randomization (friction, motor KP/KD, mass) via 'xi' parameter.
    - External YAML configuration.
    """

    def __init__(self, task_type="forward", config=None):
        super(HexapodEnv, self).__init__()

        # 1. Load configuration and basic parameters
        self.task_type = task_type
        self.config = config or {}
        self.random_enabled = self.config.get("random_enabled", True)

        # Reward weights from config or defaults
        self._ctrl_cost_weight = self.config.get("ctrl_cost_weight", 600)
        self._angle_weight = self.config.get("angle_weight", 9000)
        self._distance_weight = self.config.get("distance_weight", 90000)
        self._goal_weight = self.config.get("goal_weight", 9000)

        # Simulation flags
        self.noise = self.config.get("noise", False)
        self.std_noise = self.config.get("std_noise", 0.02)
        self.motor_model = self.config.get("motor_model", True)
        self.friction = self.config.get("friction", True)
        self.mass = self.config.get("mass", True)

        # 2. Initialize Robot Simulation Instance
        self.robo = HexapodSim.HexapodSim()

        # 3. State and Space initialization
        self._setup_spaces()
        self.reset_state_variables()

    def _setup_spaces(self):
        """Define action and observation bounds."""
        limit = math.pi / 6
        # 18 joints (3 per leg * 6 legs)
        high_act = np.ones(18, dtype=np.float32) * limit
        self.action_space = spaces.Box(-high_act, high_act, dtype=np.float32)

        # 18 joints + 2 orientation values (roll/pitch)
        high_obs = np.ones(20, dtype=np.float32) * limit
        high_obs[-1] = 90  # Pitch
        high_obs[-2] = 90  # Roll
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

    def reset_state_variables(self):
        """Internal helper to reset internal reward and tracking variables."""
        self.reward = 0.0
        self.action = np.zeros(18)
        self.obs = np.zeros(20)
        self.pos = np.zeros(3)
        self.angle = 0.0

    def reset(self, *, seed=None, options=None, xi=None):
        """
        Reset the environment with support for deterministic settings and domain randomization.
        
        Args:
            options (dict): {'angle': rad, 'obstacle_pos': [[x1,y1], [x2,y2]]}
            xi (list): [friction, kp, mass_multiplier] for domain randomization
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.reset_state_variables()
        self.robo.reset()

        # --- A. Domain Randomization (Dynamics) ---
        if xi is not None:
            if self.friction:
                self.robo.model.geom_friction[0][0] = xi[0]
            if self.motor_model:
                kp = xi[1]
                kd = kp / 10.0 + np.random.rand() * 0.1
                self.robo.pd = [HexapodSim.PDController(kp=kp, kd=kd) for _ in range(18)]
            if self.mass:
                total_mass = sum(self.robo.model.body_mass)
                self.robo.model.body_mass = (self.robo.model.body_mass / total_mass) * xi[2]

        # --- B. Environmental Layout (Task Specific) ---
        options = options or {}
        fixed_angle = options.get("angle")
        fixed_obs = options.get("obstacle_pos")

        if self.task_type == "forward":
            if fixed_obs is not None:
                self._set_fixed_obstacles(fixed_obs)
            elif self.random_enabled:
                self._randomize_obstacles()
            else:
                default_obs = self.config.get("default_obstacle_pos", [[0, 0.5], [0, 2.5]])
                self._set_fixed_obstacles(default_obs)

        elif self.task_type == "balance":
            if fixed_angle is not None:
                self._set_terrain_angle(*fixed_angle)
            elif self.random_enabled:
                self._randomize_terrain()

        # --- C. Settle Simulation ---
        for _ in range(400):
            self.robo.step(np.zeros(18))

        self.obs = self._get_obs()
        self.angle = self.robo.get_tilt_angle()
        self.pos = self.robo.get_root_pos.copy()

        return self.obs, {}

    def _set_fixed_obstacles(self, pos_list):
        """Sets deterministic positions for obstacles in the MuJoCo model."""
        self.robo.model.body_pos[-1][:2] = pos_list[0]
        self.robo.model.body_pos[-2][:2] = pos_list[1]

    def _apply_terrain_angle(self, angle_rad, yaw=0.0):
        """Applies a deterministic tilt to the floor geom."""
        self.robo.model.geom_quat[0] = np.array([
            math.cos(angle_rad / 2),
            math.sin(angle_rad / 2) * math.cos(yaw),
            math.sin(angle_rad / 2) * math.sin(yaw),
            0
        ])

    def _set_terrain_angle(self, roll, pitch):
        """Applies a deterministic tilt to the floor geom."""
        self.robo.model.geom_quat[0] = euler_to_quaternion(roll, pitch)

    def _randomize_obstacles(self):
        """Randomizes obstacles based on ranges defined in config."""
        obs_cfg = self.config.get("obstacle_range", {"x": [-0.5, 0.5], "y_near": [0.4, 0.6], "y_far": [1.5, 3.5]})
        p1 = [np.random.uniform(*obs_cfg["x"]), np.random.uniform(*obs_cfg["y_near"])]
        p2 = [np.random.uniform(*obs_cfg["x"]), np.random.uniform(*obs_cfg["y_far"])]
        self._set_fixed_obstacles([p1, p2])

    def _randomize_terrain(self):
        """Randomizes terrain slope based on ranges defined in config."""
        t_cfg = self.config.get("terrain_range", {"min_degree": 5, "max_degree": 15})
        angle = math.radians(np.random.uniform(t_cfg["min_degree"], t_cfg["max_degree"]))
        yaw = np.random.uniform(0, math.pi)
        self._apply_terrain_angle(angle, yaw)

    def _get_obs(self):
        """Retrieve state with optional Gaussian noise."""
        degrees = self.robo.get_root_degrees
        q = self.robo.get_joint_q
        state = np.concatenate([q, degrees[:2]]).astype(np.float32)

        if self.noise:
            noise = np.random.normal(0, self.std_noise, 20)
            noise[-2:] = noise[-2:] / math.pi * 6 * 90  # Scale noise for orientation
            state += noise
        return state

    def _get_rew(self):
        """Calculate task-specific reward."""
        costs = self._ctrl_cost_weight * float(np.linalg.norm(self.action - self.obs[:18]))

        if self.task_type == "forward":
            # Encourage forward progress and orientation stability
            reward_angle = (self.angle - self.robo.get_tilt_angle()) * self._angle_weight
            reward_dist = (self.robo.get_root_pos[1] - self.pos[1]) * self._distance_weight
            total_rew = reward_angle + reward_dist - costs
        else:
            # Leveling: Minimize tilt/rotation
            reward_goal = (self.angle - self.robo.get_tilt_angle) * self._goal_weight
            total_rew = reward_goal - costs

        return total_rew, {"reward_ctrl": -costs}

    def step(self, action, viewer=None):
        """Execute action with frame skipping."""
        self.action = action

        for _ in range(100):
            self.robo.step(action)
            if viewer is not None:
                viewer.sync()
                time.sleep(0.02)

        reward, info = self._get_rew()
        self.obs = self._get_obs()
        self.angle = self.robo.get_tilt_angle
        self.pos = self.robo.get_root_pos.copy()
        self.reward += reward

        # Termination Logic
        done = False
        truncated = False
        success = False

        if self.task_type == "forward":
            if self.pos[1] > 4.0:
                success = True
            if self.robo.data.time > 60:
                done = True
        else:
            if abs(self.angle) < math.radians(3):
                success = True
            if self.robo.data.time > 10:
                done = True

        return self.obs, reward, done, truncated, {"success": success, **info}


def load(task_name, config_path="./config.yaml"):
    """Factory function to load the environment."""
    config = {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using defaults.")

    return HexapodEnv(task_type=task_name, config=config)
