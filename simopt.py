import numpy as np
import Reinforcement_Env
from stable_baselines3 import DDPG
from scipy.optimize import minimize
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
    Matches the naming convention from previous trajectory recordings[cite: 436, 437].
    """
    # 1. Extract task type (forward/balance)
    task_type = "forward" if "forward" in filename.lower() else "balance"

    # 2. Extract noise setting (true/false)
    noise_match = re.search(r"noise_(true|false)", filename.lower())
    noise_enabled = noise_match.group(1) == "true" if noise_match else False

    # 3. Extract model name and load model [cite: 436]
    model_name_match = re.search(r"record-(.*?)-", filename)
    if model_name_match:
        model_name = model_name_match.group(1)
        model_dir = "./logs_ddpg_noise02/"
        model_path = os.path.join(model_dir, model_name)

        try:
            model_hexa = DDPG.load(model_path, device="cpu")
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            model_hexa = None
    else:
        print("Could not parse model name from filename.")
        model_hexa = None

    return task_type, noise_enabled, model_hexa


def load_real_stat_from_csv(csv_path):
    """
    Reads the observation trajectory from the CSV to use as ground truth[cite: 438].
    """
    df = pd.read_csv(csv_path)
    # Convert string representation of lists back to numpy arrays
    # Taking first 6 steps to match the standard simulator horizon
    obs_list = [np.fromstring(row.strip('[]'), sep=',') for row in df['obs'].head(6)]
    return np.array(obs_list)


# ===============================
# Simulator & Distance
# ===============================

def simulator(xi, task_type, noise_enabled, model_hexa):
    """
    Simulates the hexapod trajectory using the inferred task settings and loaded model[cite: 436, 437].
    """
    result = []
    # Initialize environment with task-specific settings [cite: 206]
    config = {"noise": noise_enabled}
    env_inst = Reinforcement_Env.HexapodEnv(task_type=task_type, config=config)

    # Reset with the current xi parameters (friction, mass, etc.)
    obs, _ = env_inst.reset(xi=xi)

    for i in range(6):
        action, _ = model_hexa.predict(obs, deterministic=True)
        obs, reward, done, success, info = env_inst.step(action)
        result.append(obs)

    return np.array(result)


def distance(sim_stat, real_stat, alpha_l1=0.5, alpha_l2=1):
    """Calculates weighted L1 and L2 distance between trajectories."""
    diff = sim_stat - real_stat
    l2 = np.linalg.norm(diff)
    l1 = np.linalg.norm(diff, ord=1)
    return (alpha_l2 * l2 + alpha_l1 * l1) * 0.1


# ===============================
# REPS Core Components
# ===============================

def reps_dual(eta, costs, epsilon):
    """Dual objective function for REPS optimization."""
    if eta <= 1e-8:
        return np.inf

    c_min = np.min(costs)
    stabilized = costs - c_min

    log_sum_exp = np.log(np.mean(np.exp(-stabilized / eta)))
    return eta * epsilon + eta * log_sum_exp + c_min


def solve_eta(costs, epsilon):
    """Finds the optimal lagrange multiplier eta."""
    res = minimize(
        lambda x: reps_dual(x[0], costs, epsilon),
        x0=np.array([1.0]),
        bounds=[(1e-8, 1e3)]
    )
    return res.x[0]


def update_distribution(xi_samples, costs, epsilon):
    """Updates the Gaussian distribution of xi based on REPS weights."""
    eta = solve_eta(costs, epsilon)

    c_min = np.min(costs)
    stabilized = costs - c_min

    weights = np.exp(-stabilized / eta)
    weights /= np.sum(weights)

    new_mean = np.sum(weights[:, None] * xi_samples, axis=0)

    diff = xi_samples - new_mean
    new_cov = np.einsum('i,ij,ik->jk', weights, diff, diff)
    new_cov += np.eye(new_cov.shape[0]) * 1e-6

    return new_mean, new_cov, weights, eta


# ===============================
# REPS Main Loop
# ===============================

def reps_simopt(
        real_stat,
        task_type,
        noise_enabled,
        model_hexa,
        num_samples=50,
        num_iterations=20,
        epsilon=0.5,
        init_mean=None,
        init_cov=None
):
    """
    Main REPS loop for parameter inference (Sim-to-Real optimization).
    """
    d = 3  # Dimensionality of xi [cite: 438]

    if init_mean is None:
        mean = np.array([0.8, 0.5, 0.5])
    else:
        mean = init_mean

    if init_cov is None:
        cov = np.eye(d) * 0.01
    else:
        cov = init_cov

    for t in range(num_iterations):
        start_time = time.time()
        print(f"\nIteration {t}")

        # Step 1: Sample xi from current distribution
        xi_samples = np.random.multivariate_normal(mean, cov, size=num_samples)
        xi = np.clip(xi, [0.1, 0.1, 0.1], [2.0, 2.0, 2.0])

        # Step 2: Simulate and calculate costs
        costs = []
        for xi in xi_samples:
            sim_stat = simulator(xi, task_type, noise_enabled, model_hexa)
            dist = distance(sim_stat, real_stat)
            costs.append(dist)

        costs = np.array(costs)

        print(f"Mean cost: {np.mean(costs):.4f}")
        print(f"Min cost: {np.min(costs):.4f}")

        # Step 3: Update distribution via REPS
        mean, cov, weights, eta = update_distribution(xi_samples, costs, epsilon)

        print("Updated Mean Xi:", mean)
        print("Iteration runtime:", time.time() - start_time)

    return mean, cov


# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    np.random.seed(42)

    # Path to the recorded performance CSV
    csv_filename = "ddpg_trajectory_record-ddpg_495000_steps-forward-noise_false.csv"

    if os.path.exists(csv_filename):
        # 1. Parse task settings and load model
        task, noise, model = parse_filename_settings(csv_filename)

        if model is not None:
            # 2. Load real trajectory from CSV
            real_trajectory = load_real_stat_from_csv(csv_filename)

            # 3. Run REPS Optimization
            final_mean, final_cov = reps_simopt(
                real_stat=real_trajectory,
                task_type=task,
                noise_enabled=noise,
                model_hexa=model,
                num_samples=80,
                num_iterations=15,
                epsilon=0.5
            )

            print("\nOptimization Complete.")
            print("Final Parameter Mean (Xi):", final_mean)
            print("Final Parameter Covariance:", final_cov)
    else:
        print(f"Error: {csv_filename} not found.")