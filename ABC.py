import numpy as np
import Reinforcement_Env
from stable_baselines3 import DDPG
import time
import pandas as pd
import re
import os


# ===============================
# Configuration & Helper Functions
# ===============================

def parse_filename_settings(filename):
    """
    Extracts task_type, noise settings, and loads the corresponding model from the CSV filename.
    Example filename: ddpg_trajectory_record-ddpg_495000_steps-forward-noise_false.csv
    """
    # 1. Extract task type (forward/balance)
    # Using 'level' or 'balance' logic based on your previous logs
    task_type = "forward" if "forward" in filename.lower() else "balance"

    # 2. Extract noise setting (true/false)
    noise_match = re.search(r"noise_(true|false)", filename.lower())
    noise_enabled = noise_match.group(1) == "true" if noise_match else False

    # 3. Extract model name and load model
    # Regex looks for the string between 'record-' and the next '-'
    # e.g., extracts 'ddpg_495000_steps'
    model_name_match = re.search(r"record-(.*?)-", filename)
    if model_name_match:
        model_name = model_name_match.group(1)

        # Define your model directory (update this path to your actual logs folder)
        model_dir = "./logs_ddpg_noise02/"
        model_path = os.path.join(model_dir, model_name)

        try:
            # Load the DDPG model
            model_hexa = DDPG.load(model_path, device="cpu")
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            model_hexa = None
    else:
        print("Could not parse model name from filename.")
        model_hexa = None

    # Return all three values
    return task_type, noise_enabled, model_hexa


def load_real_stat_from_csv(csv_path):
    """
    Reads the observation trajectory from the CSV to use as real_stat.
    Assumes 'obs' column contains string representations of lists.
    """
    df = pd.read_csv(csv_path)
    # Convert string representation of list back to numpy arrays
    # Taking first 6 steps to match the simulator's horizon
    obs_list = [np.fromstring(row.strip('[]'), sep=',') for row in df['obs'].head(10)]
    return np.array(obs_list)


# ===============================
# Simulator Section
# ===============================

def simulator(xi, task_type, noise_enabled, model_hexa):
    """
    Inputs:
        xi: (3,) dynamic parameters [friction, kp, mass_multiplier]
    Outputs:
        result: List of observations across the trajectory
    """
    runtime = 10
    result = []
    # Initialize environment with settings derived from filename
    config = {"noise": noise_enabled}
    env_inst = Reinforcement_Env.HexapodEnv(task_type=task_type, config=config)

    # Reset with the current xi (domain randomization parameters)
    obs, _ = env_inst.reset(xi=xi)

    for i in range(runtime):
        if model_hexa is None:
            raise ValueError("Model not loaded. Check filename parsing.")
        action, _ = model_hexa.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_inst.step(action)
        result.append(obs)

    env_inst.close()

    return np.array(result)


def distance(sim_stat, real_stat):
    """Calculates Euclidean distance between simulated and real trajectories."""
    return np.linalg.norm(sim_stat - real_stat)


# ===============================
# ABC-SMC Main Algorithm
# ===============================

def abc_smc(real_stat, task_type, noise_enabled, model_hexa, num_particles=30, num_generations=5, eps_schedule=None,
            perturb_scale=0.02):
    """
    Approximate Bayesian Computation - Sequential Monte Carlo
    Returns: particles (N, d), weights (N,)
    """
    d = 3  # Dimensionality of xi
    if eps_schedule is None:
        eps_schedule = np.linspace(16, 13, num_generations)

    # Initial prior sampling (uniform or specific starting point)
    particles = []
    weights = []

    # Output file for logging results
    with open("abc_smc_results.txt", "w") as f:
        f.write(f"Starting ABC-SMC for Task: {task_type}, Noise: {noise_enabled}\n")

        for t in range(num_generations):
            eps = eps_schedule[t]
            new_particles = []
            new_weights = []
            start_time = time.time()

            print(f"Generation {t}, epsilon={eps:.3f}")

            while len(new_particles) < num_particles:
                if t == 0:
                    # Initial Generation: Sample from a broad prior
                    xi = np.array([np.random.uniform(0.5, 1.5),  # Friction
                                   np.random.uniform(0.1, 1.0),  # KP
                                   np.random.uniform(0.5, 2.0)])  # Mass
                else:
                    # Subsequent Generations: Sample from previous particles + perturbation
                    idx = np.random.choice(len(particles), p=weights)
                    xi = particles[idx] + perturb_scale * np.random.randn(d)
                    xi = np.clip(xi, [0.1, 0.1, 0.1], [2.0, 2.0, 3.0])

                sim_stat = simulator(xi, task_type, noise_enabled, model_hexa)
                dist = distance(sim_stat, real_stat)

                if dist < eps:
                    new_particles.append(xi)
                    if t == 0:
                        new_weights.append(1.0)
                    else:
                        # Importance weighting
                        denom = 0.0
                        for j in range(len(particles)):
                            diff = xi - particles[j]
                            kernel = np.exp(-0.5 * np.sum(diff ** 2) / perturb_scale ** 2)
                            denom += weights[j] * kernel
                        new_weights.append(1.0 / max(denom, 1e-6))

            # Normalize weights
            new_weights = np.array(new_weights)
            new_weights /= np.sum(new_weights)
            particles = np.array(new_particles)
            weights = new_weights

            # Statistics calculation
            mean_xi = np.average(particles, axis=0, weights=weights)
            diff = particles - mean_xi
            cov_xi = np.einsum('i,ij,ik->jk', weights, diff, diff)

            # Log to file
            log_str = f"\nGeneration {t} completed in {time.time() - start_time:.2f}s\n"
            log_str += f"Mean Xi: {mean_xi}\n"
            log_str += f"Covariance: \n{cov_xi}\n"
            f.write(log_str)
            f.flush()

    return particles, weights


# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Define the path to your recorded CSV
    csv_filename = "ddpg_trajectory_record-ddpg_495000_steps-forward-noise_false.csv"

    if os.path.exists(csv_filename):
        # 2. Parse settings from filename
        task, noise, model = parse_filename_settings(csv_filename)

        # 3. Load real statistics from the CSV
        real_trajectory = load_real_stat_from_csv(csv_filename)

        # 4. Run ABC-SMC
        final_particles, final_weights = abc_smc(
            real_stat=real_trajectory,
            task_type=task,
            noise_enabled=noise,
            num_particles=50,
            num_generations=10,
            model_hexa=model
        )

        print("Inference complete. Check abc_smc_results.txt for details.")
    else:
        print(f"Error: {csv_filename} not found.")
