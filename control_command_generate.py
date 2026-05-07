from stable_baselines3 import DDPG
import numpy as np
import math


def convert(action):
    """
    Converts RL agent output (radians) into hardware-specific
    PWM commands and formats them into serial strings.
    """
    # Hardware IDs for the hexapod servos
    index = ["011", "012", "013", "014", "015", "016",
             "021", "022", "023", "024", "025", "026",
             "031", "032", "033", "034", "035", "036"]

    # Mapping RL actions to physical motor indices and directions
    command = np.zeros(18)
    command[0] = -action[5]
    command[1] = -action[4]
    command[2] = -action[3]
    command[3] = -action[0]
    command[4] = action[1]
    command[5] = action[2]
    command[6] = action[14]
    command[7] = action[13]
    command[8] = -action[12]
    command[9] = -action[15]
    command[10] = -action[16]
    command[11] = -action[17]
    command[12] = -action[8]
    command[13] = -action[7]
    command[14] = -action[6]
    command[15] = -action[9]
    command[16] = action[10]
    command[17] = action[11]

    # Convert radians to PWM pulse width (standard 500-2500 range, 1500 as center)
    command = [int(rad / math.pi * 2000 + 1500) for rad in command]

    # Group commands into three packets to optimize Bluetooth transmission
    def build_packet(start_idx, end_idx):
        packet = "{"
        for i in range(start_idx, end_idx):
            packet += f"#{index[i]}P{command[i]}T0800!"
        packet += "}"
        return packet

    return [build_packet(0, 6), build_packet(6, 12), build_packet(12, 18)]


class Agent:
    def __init__(self):
        # State and action buffers
        self.obs = np.zeros(20)
        self.action = np.zeros(18)
        self.undo_action = np.zeros(18)  # Stores predicted but unconfirmed action

        # Tilt/Angle tracking for balance calculations
        self.theta = 0
        self.curr_theta = 0

        # Load the pre-trained DDPG model (Balance mode)
        # Using CPU to ensure compatibility with real-time control loops
        self.model = DDPG.load("./ddpg/ddpg_500000_steps.zip", device="cpu")

    def step(self, s1, s2):
        """
        Processes sensor input, predicts next action, and returns formatted commands.
        s1, s2: Sensor inputs (typically tilt angles in degrees)
        """
        # Calculate current tilt magnitude
        rad_s1, rad_s2 = np.deg2rad(s1), np.deg2rad(s2)
        self.curr_theta = np.arctan(np.sqrt(np.tan(rad_s1) ** 2 + np.tan(rad_s2) ** 2))

        # Calculate balance cost for debugging/monitoring
        delt_angle = np.rad2deg(self.theta - self.curr_theta)
        print(f"Balance cost: {delt_angle * 9000:.4f}")

        # Construct observation: Previous actions + current sensor readings
        obs = np.append(self.action, [s1, s2])

        # Predict next action using the DDPG agent
        action, _ = self.model.predict(obs, deterministic=True)
        self.undo_action = action

        return convert(action)

    def manual(self, action):
        """Allows for manual override of motor positions"""
        self.action = action
        return convert(action)

    def send(self):
        """
        Confirms the execution of the action.
        Updates the internal state and logs control effort.
        """
        control_cost = np.linalg.norm(self.action - self.undo_action) * 600
        print(f"Control cost: {control_cost:.4f}")

        # Shift predicted action to current action and update theta reference
        self.action = self.undo_action
        self.theta = self.curr_theta
