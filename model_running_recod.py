import Reinforcement_Env
import time
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
import os

# --- Configuration Area ---
MODEL_PATH = ""
N_STEPS = 200  # Duration of the recording (Number of steps)
ENV_SETTINGS = {
    "task_type": "forward",
    "noise_enabled": False,
    "friction_enabled": False,
    "motor_enabled": False,
}


def build_config_string(config: dict):
    parts = []
    for k, v in config.items():
        if isinstance(v, bool):
            v = str(v).lower()
        parts.append(f"{k}_{v}")
    return "-".join(parts)

def record_agent_performance():
    # 1. Initialize environment
    test_env = Reinforcement_Env.HexapodEnv(
    task_type=ENV_SETTINGS["task_type"],
    noise_enabled=ENV_SETTINGS["noise_enabled"],
    friction_enabled=ENV_SETTINGS["friction_enabled"],
    motor_enabled=ENV_SETTINGS["motor_enabled"],
)

    # 2. Load model and extract filename for the CSV prefix
    # os.path.splitext(os.path.basename(...))[0] gets 'ddpg_495000_steps' from the path
    model_file_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    model = DDPG.load(MODEL_PATH, device="cpu")

    # 3. Construct dynamic CSV filename
    config_str = build_config_string(ENV_SETTINGS)

    csv_save_path = f"ddpg_trajectory_record-{model_file_name}-{config_str}.csv"


    # 4. Setup Mujoco viewer
    viewer = test_env.robo.my_render()

    # 5. Prepare data container
    data_log = {
        "step": [],
        "reward": [],
        "pos_x": [], "pos_y": [], "pos_z": [],
        "roll": [], "pitch": [], "yaw": [],
        "foot_contact": [],
        "actions": []
    }

    # 6. Reset environment
    obs, _ = test_env.reset()
    print(f"Recording started. Saving to: {csv_save_path}")

    try:
        for i in range(N_STEPS):
            # Model prediction
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, truncated, info = test_env.step(action, viewer)

            # Get states from simulation
            current_pos = test_env.robo.get_pos
            current_degrees = test_env.robo.get_degrees

            # Record touch sensors (first 6 values for 6 feet)
            contacts = test_env.robo.data.sensordata[:6].copy()

            # Log data
            data_log["step"].append(i)
            data_log["reward"].append(reward)
            data_log["pos_x"].append(current_pos[0])
            data_log["pos_y"].append(current_pos[1])
            data_log["pos_z"].append(current_pos[2])
            data_log["roll"].append(current_degrees[0])
            data_log["pitch"].append(current_degrees[1])
            data_log["yaw"].append(current_degrees[2])
            data_log["foot_contact"].append(contacts.tolist())
            data_log["actions"].append(action.tolist())

            viewer.sync()
            time.sleep(0.01)

            if done or truncated:
                print(f"Trajectory ended early at step {i}")
                break

    finally:
        # 7. Save to CSV
        df = pd.DataFrame(data_log)
        df.to_csv(csv_save_path, index=False)

        print("-" * 30)
        print(f"Task: {ENV_SETTINGS['task_type']}")
        print(f"Noise: {ENV_SETTINGS['noise_enabled']}")
        print(f"Result saved to: {csv_save_path}")


if __name__ == "__main__":
    record_agent_performance()
