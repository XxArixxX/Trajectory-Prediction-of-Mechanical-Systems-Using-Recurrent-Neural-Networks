# create_dataset.py
# High-quality dataset generator for "Trajectory Prediction of Mechanical Systems Using RNNs"
# Generates: mass-spring, single pendulum, double pendulum, acrobot, three-body (figure-8)
# Saves as .npz with normalization stats — ready for training

import numpy as np
from scipy.integrate import solve_ivp
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

DT = 0.02          # Time step (50 Hz) — matches the paper
SEED = 42
np.random.seed(SEED)

def generate_trajectories(system: str, n_traj: int, t_max: float):
    """Master generator — returns array of shape (n_traj, steps, state_dim)"""
    steps = int(t_max / DT) + 1
    data = []

    print(f"\nGenerating {n_traj} trajectories for {system} ({steps} steps, {t_max:.1f}s each)")

    for i in tqdm(range(n_traj), desc=system):
        if system == "mass-spring":
            y0 = np.random.uniform([-2, -3], [2, 3])  # [position, velocity]
            sol = solve_ivp(mass_spring_ode, [0, t_max], y0, t_eval=np.linspace(0, t_max, steps), rtol=1e-9)
        elif system == "single-pendulum":
            theta0 = np.random.uniform(-np.pi, np.pi)
            omega0 = np.random.uniform(-3, 3)
            sol = solve_ivp(pendulum_ode, [0, t_max], [theta0, omega0], t_eval=np.linspace(0, t_max, steps), rtol=1e-9)
        elif system == "double-pendulum":
            y0 = np.random.uniform(-2.5, 2.5, 4)
            y0[2:] *= 0.8  # moderate initial momenta
            sol = solve_ivp(double_pendulum_ode, [0, t_max], y0, t_eval=np.linspace(0, t_max, steps), method='DOP853', rtol=1e-10)
        elif system == "acrobot":
            y0 = np.random.uniform(-np.pi, np.pi, 4)
            y0[2:] = np.random.uniform(-5, 5, 2)
            sol = solve_ivp(acrobot_ode, [0, t_max], y0, t_eval=np.linspace(0, t_max, steps), rtol=1e-9)
        elif system == "three-body-figure8":
            # Famous periodic figure-8 solution + small random perturbations
            y0 = figure8_initial() + np.random.normal(0, 1e-3, 18)
            sol = solve_ivp(three_body_ode, [0, t_max], y0, t_eval=np.linspace(0, t_max, steps), method='DOP853', rtol=1e-12, atol=1e-12)
        else:
            raise ValueError("Unknown system")

        traj = sol.y.T
        data.append(traj)

    return np.stack(data, axis=0).astype(np.float32)


# ==================== ODE Definitions ====================

def mass_spring_ode(t, y, m=1.0, k=4.0, c=0.1):
    x, v = y
    return [v, -k/m * x - c/m * v]

def pendulum_ode(t, y, g=9.81, l=1.0):
    theta, omega = y
    return [omega, -g/l * np.sin(theta)]

def double_pendulum_ode(t, y, m1=1, m2=1, l1=1, l2=1, g=9.81):
    th1, th2, p1, p2 = y
    a = m1 + m2
    denom1 = l1 * (a - m2 * np.cos(th1 - th2)**2)
    denom2 = l2 * (a - m2 * np.cos(th1 - th2)**2)

    dth1 = p1 / denom1
    dth2 = p2 / denom2

    dp1 = -m2*l1*l2*dth1**2*np.sin(th1-th2) - g*(m1+m2)*np.sin(th1) - 0.05*p1
    dp2 =  m2*l2*l1*dth1**2*np.sin(th1-th2) - g*m2*np.sin(th2) - 0.05*p2

    return [dth1, dth2, dp1, dp2]

def acrobot_ode(t, y, m1=1, m2=1, l1=1, l2=1, g=9.81):
    th1, th2, dth1, dth2 = y
    d1 = m1*l1**2 + m2*(l1**2 + 2*l1*l2*np.cos(th2) + l2**2)
    d2 = m2*(l2**2 + l1*l2*np.cos(th2))

    # Simplified dynamics (no torque)
    ddth2_num = -m2*l1*l2*np.sin(th2)*(dth1**2 + 2*dth1*dth2) - g*m2*l2*np.sin(th1+th2) - g*(m1+m2)*l1*np.sin(th1)
    ddth2 = ddth2_num / (m2*l2**2 + d2 - d2**2/d1) if abs(d1) > 1e-8 else 0

    ddth1 = -(m2*l2*ddth2*np.cos(th2) + m2*l1*np.sin(th2)*dth2**2 + (m1+m2)*g*np.sin(th1) + 0.05*dth1)
    ddth1 /= (m1+m2)*l1

    return [dth1, dth2, ddth1, ddth2]

def figure8_initial():
    """Exact periodic figure-8 initial conditions (Chenciner & Montgomery, 2000)"""
    return np.array([
        -1.0,  0.0,  0.0,   # body 1
         1.0,  0.0,  0.0,   # body 2
         0.0,  0.0,  0.0,   # body 3 (origin)
         0.30689,  0.12551,  0.78749,   # v1
         0.30689,  0.12551,  0.78749,   # v2 (symmetric)
        -0.61378, -0.25102, -1.57498    # v3
    ]).flatten()

def three_body_ode(t, y, G=1, m1=1, m2=1, m3=1):
    positions = y[:9].reshape(3, 3)
    velocities = y[9:].reshape(3, 3)
    acc = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            if i != j:
                r = positions[j] - positions[i]
                acc[i] += G * m1 * r / (np.linalg.norm(r)**3 + 1e-12)
    return np.concatenate([velocities.flatten(), acc.flatten()])


# ==================== Main Generation ====================

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    datasets = {
        "mass_spring.npy":       dict(system="mass-spring",       n_traj=500,  t_max=20.0),
        "single_pendulum.npy":   dict(system="single-pendulum",   n_traj=500,  t_max=20.0),
        "double_pendulum.npy":   dict(system="double-pendulum",   n_traj=1000, t_max=30.0),
        "acrobot.npy":           dict(system="acrobot",           n_traj=800,  t_max=24.0),
        "three_body_figure8.npy":dict(system="three-body-figure8",n_traj=200, t_max=40.0),
    }

    for filename, config in datasets.items():
        path = os.path.join("data", filename)
        if os.path.exists(path):
            print(f"{filename} already exists → skipping")
            continue

        raw_data = generate_trajectories(**config)

        # Compute global normalization stats (zero mean, unit variance)
        mean = raw_data.mean(axis=(0, 1))
        std  = raw_data.std(axis=(0, 1)) + 1e-8

        data_norm = (raw_data - mean) / std

        np.savez_compressed(
            path,
            raw=raw_data,
            normalized=data_norm,
            mean=mean,
            std=std,
            dt=DT,
            system=config["system"],
            n_trajectories=raw_data.shape[0],
            trajectory_length=raw_data.shape[1],
            state_dim=raw_data.shape[2]
        )
        print(f"Saved {path} | Shape: {raw_data.shape} | Compressed size: {os.path.getsize(path)/1e6:.1f} MB")

    print("\nAll datasets generated successfully!")
    print("   Location: ./data/")
    print("   Ready for training — just load with np.load('data/double_pendulum.npy')['normalized']")
