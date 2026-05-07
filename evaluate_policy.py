import os
import csv
import Reinforcement_Env
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# --- Configuration Section ---
# Set these to match the environment settings used during training
env_settings = {
    "motor": True,
    "mass": True,
    "friction": True,
    "noise": False
}


def get_dynamic_filename(prefix="output_ddpg", ext=".csv"):
    """
    Generates a filename based on active environment settings.
    Example: output_ddpg_motor_mass_friction.csv
    """
    # Extract keys where value is True
    active_params = [k for k, v in env_settings.items() if v is True]
    suffix = "_".join(active_params)
    return f"{prefix}_{suffix}{ext}" if suffix else f"{prefix}_default{ext}"


def write_results(file_path, model_name, mean_reward, std_reward):
    """Logs evaluation results to the dynamically named CSV file."""
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Model_Name', 'Mean_Reward', 'Std_Reward'])
        writer.writerow([model_name, mean_reward, std_reward])


def evaluate_folder(folder_path, n_eval_episodes=20):
    """Iterates through models in a folder and evaluates their performance."""
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return

    # Determine the output file path based on settings
    output_file = get_dynamic_filename()
    print(f"Results will be saved to: {output_file}")

    # List all .zip model files
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.zip')]

    for i, item in enumerate(filenames):
        print(f"[{i + 1}/{len(filenames)}] Processing: {item}")

        # Identify task type from filename (assuming 'forward' or 'balance' is in the name)
        current_task = "forward" if "forward" in item.lower() else "balance"

        # Initialize vectorized environment with specific config
        env_hexa = make_vec_env(lambda: Reinforcement_Env.HexapodEnv(
            task_type=current_task,
            config=env_settings
        ))

        try:
            # Load the model 
            model_full_path = os.path.join(folder_path, item)
            model = DDPG.load(model_full_path, env=env_hexa)

            # Perform evaluation
            mean_reward, std_reward = evaluate_policy(
                model,
                env_hexa,
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            )

            # Log results
            write_results(output_file, item, mean_reward, std_reward)
            print(f"   Result: {mean_reward:.2f} +/- {std_reward:.2f}")

        except Exception as e:
            print(f"   Error evaluating {item}: {e}")

        finally:
            # Close environment to free MuJoCo resources
            env_hexa.close()


if __name__ == "__main__":
    evaluate_folder()