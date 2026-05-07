import numpy as np
import pandas as pd
import Reinforcement_Env
from stable_baselines3 import DDPG
from scipy.stats import entropy
import time
import os
import re


# ===============================
# Configuration & Helper Functions
# ===============================

def parse_filename_settings(filename):
    """
    Extracts task_type, noise settings, and loads the corresponding model from the CSV filename.
    """
    task_type = "forward" if "forward" in filename.lower() else "balance"
    noise_match = re.search(r"noise_(true|false)", filename.lower())
    noise_enabled = noise_match.group(1) == "true" if noise_match else False

    model_name_match = re.search(r"record-(.*?)-", filename)
    model_name = model_name_match.group(1) if model_name_match else "default_model"

    # Adjust model_dir as per your local setup
    model_dir = "./logs_ddpg_noise02/"
    model_path = os.path.join(model_dir, model_name)

    try:
        model_hexa = DDPG.load(model_path, device="cpu")
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        model_hexa = None

    return task_type, noise_enabled, model_hexa


def load_real_data_from_csv(csv_path):
    """
    Extracts multi-modal data from CSV: rewards, posture (roll, pitch), and foot contact.
    """
    df = pd.read_csv(csv_path)

    # Extracting components for realword data distribution
    real_rewards = df['reward'].values

    # Convert string lists back to numpy arrays
    real_posture = np.array([np.fromstring(row.strip('[]'), sep=',') for row in df['euler']])
    real_foot_contact = np.array([np.fromstring(row.strip('[]'), sep=',') for row in df['foot_contact']])

    # Combine into a dictionary for distribution comparison
    return {
        "rewards": real_rewards,
        "posture": real_posture,
        "foot_contact": real_foot_contact
    }


# =========================
# Metric Functions
# =========================

def js_divergence(p, q, eps=1e-8):
    """Calculates Jensen-Shannon divergence between two distributions."""
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def compute_distribution(data, bins=30, data_range=None):
    """Computes density histogram for a given dataset."""
    hist, _ = np.histogram(data, bins=bins, range=data_range, density=True)
    hist = hist + 1e-8
    hist = hist / np.sum(hist)
    return hist


# =========================
# Simulation Section
# =========================

def simulate(xi, model, task_type, noise_enabled):
    """
    Runs simulation using xi parameters and specific terrain angles.
    Redundancy removed: euler_to_quaternion is now handled by env.reset(options).
    """
    # Initial terrain conditions (roll, pitch) recorded from real world
    real_init_angles = [
        [-8.0, -2.1], [2.2, -15.2], [10.7, -1.3],
        [-0.6, 15.4], [8.8, -5.8], [12.8, 4.1]
    ]

    sim_results = {"rewards": [], "posture": [], "foot_contact": []}
    config = {"noise": noise_enabled}

    for angles in real_init_angles:
        env_inst = Reinforcement_Env.HexapodEnv(task_type=task_type, config=config)

        # Integration: Pass terrain angles directly to reset options [cite: 660]
        obs, _ = env_inst.reset(xi=xi, options={"angle_roll": angles[0], "angle_pitch": angles[1]})

        for _ in range(10):  # Match simulation horizon
            if model_hexa is None:
                raise ValueError("Model not loaded. Check filename parsing.")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env_inst.step(action)

            # Assuming env_inst provides these or they are derived from obs/info
            sim_results["rewards"].append(reward)
            sim_results["posture"].append(info.get('euler', np.zeros(2)))
            sim_results["foot_contact"].append(info.get('foot_contact', np.zeros(6)))

    return sim_results


# ===============================
# TCBI Algorithm
# ===============================

def tcbi(real_data, model, task_type, noise_enabled, num_particles=30, num_generations=5):
    """
    using multi-modal distribution distance.
    """

    def gaussian_kernel(x, mu, sigma=0.02):
    return np.exp(-0.5 * np.sum((x - mu)**2) / sigma**2)

    d = 3  # [friction, kp, mass_multiplier]
    eps_schedule = np.linspace(8, 5, num_generations)

    # Compute real distributions
    P_real_rew = compute_distribution(real_data["rewards"])
    P_real_pos = compute_distribution(real_data["posture"].flatten())
    P_real_ft = compute_distribution(real_data["foot_contact"].flatten())

    particles = []
    weights = []

    for t in range(num_generations):
        eps = eps_schedule[t]
        new_particles = []
        new_weights = []

        print(f"\n--- Generation {t}, epsilon={eps:.3f} ---")

        while len(new_particles) < num_particles:
            if t == 0:
                xi = np.array([np.random.uniform(0.6, 1.2), np.random.uniform(0.2, 0.8), np.random.uniform(0.5, 1.5)])
            else:
                idx = np.random.choice(len(particles), p=weights)
                xi = particles[idx] + 0.02 * np.random.randn(d)
                xi = np.clip(xi, [0.5, 0.1, 0.3], [1.5, 1.0, 2.0])

            sim_data = simulate(xi, model, task_type, noise_enabled)

            # Compute distance across all distributions
            dist_rew = js_divergence(P_real_rew, compute_distribution(sim_data["rewards"]))
            dist_pos = js_divergence(P_real_pos, compute_distribution(np.array(sim_data["posture"]).flatten()))
            dist_ft = js_divergence(P_real_ft, compute_distribution(np.array(sim_data["foot_contact"]).flatten()))

            total_dist = (
                0.5 * dist_rew +
                0.3 * dist_pos +
                0.2 * dist_ft
            )

            if total_dist < eps:
                new_particles.append(xi)
                prior = 1.0  # uniform prior
                denom = np.sum([
                    weights[j] * gaussian_kernel(xi, particles[j])
                    for j in range(len(particles))
                ]) + 1e-8

                weight = prior / denom
                new_weights.append(weight)

        particles = np.array(new_particles)
        weights = np.array(new_weights) / np.sum(new_weights)

    return particles, weights


if __name__ == "__main__":
    csv_file = "ddpg_trajectory_record-ddpg_495000_steps-forward-noise_false.csv"

    if os.path.exists(csv_file):
        task, noise, model_inst = parse_filename_settings(csv_file)
        real_data_map = load_real_data_from_csv(csv_file)

        final_particles, final_weights = tcbi(
            real_data=real_data_map,
            model=model_inst,
            task_type=task,
            noise_enabled=noise,
            num_particles=50,
            num_generations=10
        )
        print("Inference Complete. Mean Xi:", np.average(final_particles, axis=0, weights=final_weights))
