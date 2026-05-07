import Reinforcement_Env
import os
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# 1. Configuration Setup
# Modify these values to change the task and environment physics
config = {
    "task_type": "forward",  # Options: "forward", "balance"
    "motor": True,
    "mass": True,
    "friction": True,
    "noise": False
}

# 2. Dynamic Path and Prefix Generation
# Extract active physical features (True values) to build the folder name
active_features = [key for key, value in config.items() if value is True and key != "task_type"]
feature_suffix = "_".join(active_features) if active_features else "vanilla"

# Final path and prefix strings
# e.g., logs_ddpg_motor_mass_friction/
save_path = f'logs_ddpg_{feature_suffix}/'
# e.g., ddpg_forward
name_prefix = f'ddpg_{config["task_type"]}'

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# 3. Environment Initialization
# Pass task_type and config to your HexapodEnv
env1 = make_vec_env(
    lambda: Reinforcement_Env.HexapodEnv(
        task_type=config["task_type"],
        config=config
    )
)

# 4. Model Definition
policy_kwargs = dict(net_arch=[256, 256])
model = DDPG("MlpPolicy", env1, verbose=1, device="cuda", policy_kwargs=policy_kwargs)

# 5. Callback for saving models
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=save_path,
    name_prefix=name_prefix
)

# Logging start info for clarity
print(f"Starting experiment: {name_prefix}")
print(f"Storage path: {save_path}")

# 6. Start Training
model.learn(total_timesteps=300000, callback=checkpoint_callback)