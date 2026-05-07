import math
import numpy as np
import mujoco as mj
import mujoco.viewer

# Path to the scene file (ensure hexapod.xml is in the same directory)
ROBO_FILENAME = "./Hexapod_XML_Model/scene.xml"


def quat_mul(q1, q2):
    """Multiplies two quaternions [w, x, y, z]."""
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return np.array([a, b, c, d])


def quaternion_to_degrees(w, x, y, z):
    """Converts quaternion [w, x, y, z] to Euler angles (degrees)."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return [math.degrees(angle) for angle in [roll_x, pitch_y, yaw_z]]


class PDController:
    """Discrete-time Proportional-Derivative Controller."""

    def __init__(self, kp, kd, dt=0.01):
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.prev_error = 0.0

    def compute(self, target, current):
        error = target - current
        # Discrete-time derivative approximation
        error_derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.kd * error_derivative


class HexapodSim:
    def __init__(self):
        # Load model and data
        try:
            self.model = mj.MjModel.from_xml_path(filename=ROBO_FILENAME)
            self.data = mj.MjData(self.model)
        except Exception as e:
            print(f"Error loading XML: {e}")
            raise

        # Initialize PD controllers for 18 joints (3 per leg for 6 legs)
        # dt is aligned with the recommended simulation timestep
        self.pd_controllers = [PDController(kp=0.1, kd=0.05, dt=self.model.opt.timestep) for _ in range(18)]

    def reset(self):
        """Resets simulation data and controller memory."""
        mj.mj_resetData(self.model, self.data)
        for pdc in self.pd_controllers:
            pdc.prev_error = 0
        return self._get_obs()

    def step(self, action=None):
        """
        Advances the simulation by one timestep.
        action: array of 18 target joint positions.
        """
        if action is None:
            action = np.zeros(self.model.nu)

        # Get current joint positions (last 18 values of qpos)
        current_joint_pos = self.get_joint_q

        # Compute control signals for each actuator
        control_signals = []
        for i, pdc in enumerate(self.pd_controllers):
            control_signals.append(pdc.compute(action[i], current_joint_pos[i]))

        self.data.ctrl = np.array(control_signals)
        mj.mj_step(self.model, self.data)

        return self._get_obs()

    def _get_obs(self):
        """Returns the current state dictionary."""
        return {
            "q_pos": self.data.qpos.copy().tolist(),
            "q_vel": self.data.qvel.copy().tolist(),
            "ctrl": self.data.ctrl.copy().tolist(),
        }

    def launch_viewer(self):
        """Launches a passive visualizer with contact force debugging enabled."""
        viewer = mj.viewer.launch_passive(self.model, self.data)
        # Enable contact force visualization for debugging foot interaction
        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.model.vis.map.force = 0.03
        return viewer

    @property
    def get_root_pos(self):
        """Returns the [x, y, z] position of the torso."""
        return self.data.qpos[:3]

    @property
    def get_root_quat(self):
        """Returns the [w, x, y, z] orientation of the torso."""
        return self.data.qpos[3:7]

    @property
    def get_joint_q(self):
        """Returns the current angles for the 18 leg joints."""
        return self.data.qpos[-18:]

    @property
    def get_root_degrees(self):
        """Returns the torso orientation in Euler angles (degrees)."""
        return quaternion_to_degrees(*self.get_root_quat)

    @property
    def get_tilt_angle(self):
        """Calculates the tilt angle of the robot relative to the world Z-axis."""
        q = self.get_root_quat
        v_local = np.array([0, 0, 0, 1])  # Local Z-vector
        q_conj = q * [1, -1, -1, -1]

        # Rotate local vector to world frame: q * v * q_conj
        tmp = quat_mul(q, v_local)
        v_world_quat = quat_mul(tmp, q_conj)
        v_world = v_world_quat[1:]

        # Compute angle between world Z and current torso Z
        cos_theta = v_world[2] / (np.linalg.norm(v_world) + 1e-9)
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
