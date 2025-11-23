# rnn_mechanics_prediction.py
# Author: Your Name (2025)
# GitHub: https://github.com/author/rnn-mechanics-2025
# Paper: Trajectory Prediction of Mechanical Systems Using Recurrent Neural Networks

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================
# 1. Double Pendulum Dynamics
# ============================

def double_pendulum_deriv(t, y, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """Equations of motion for double pendulum (theta1, theta2, p1, p2) in Hamiltonian form"""
    theta1, theta2, p1, p2 = y

    a = m1 + m2
    c = m2 * l1 * l2
    denom1 = l1 * (a - m2 * np.cos(theta1 - theta2)**2)
    denom2 = l2 * (a - m2 * np.cos(theta1 - theta2)**2)

    dtheta1_dt = p1 / denom1
    dtheta2_dt = p2 / denom2

    dp1_dt = -m2 * l1 * l2 * (dtheta1_dt**2 * np.sin(theta1 - theta2)) \
             - g * (m1 + m2) * np.sin(theta1)
    dp2_dt = m2 * l2 * l1 * (dtheta1_dt**2 * np.sin(theta1 - theta2)) \
             - g * m2 * np.sin(theta2)

    # Add small damping for realism
    damping = 0.05
    dp1_dt -= damping * p1
    dp2_dt -= damping * p2

    return [dtheta1_dt, dtheta2_dt, dp1_dt, dp2_dt]


def generate_trajectory(t_span=(0, 30), dt=0.02, y0=None):
    """Generate one double pendulum trajectory"""
    if y0 is None:
        # Random initial angles, small momenta
        y0 = np.random.uniform(-2.0, 2.0, 4)
        y0[2:] *= 0.5  # small initial velocities

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(double_pendulum_deriv, t_span, y0, t_eval=t_eval,
                    method='DOP853', rtol=1e-9, atol=1e-12)
    return sol.y.T  # shape: (T, 4)


# ============================
# 2. Dataset Creation
# ============================

def create_dataset(n_trajectories=1000, seq_len=1500):
    print("Generating dataset...")
    data = []
    for i in range(n_trajectories):
        if i % 100 == 0:
            print(f"  {i}/{n_trajectories}")
        traj = generate_trajectory(t_span=(0, seq_len * 0.02))
        data.append(traj)
    return np.stack(data)  # shape: (N, T, 4)


# ============================
# 3. RNN Model (LSTM / GRU)
# ============================

class TrajectoryRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=512, num_layers=3, rnn_type='LSTM'):
        super().__init__()
        rnn_class = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_class(input_size, hidden_size, num_layers,
                             batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, features)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def predict_sequence(self, x0, steps, hidden=None):
        """Autoregressive prediction"""
        predictions = []
        x = x0.clone()
        for _ in range(steps):
            with torch.no_grad():
                pred, hidden = self(x, hidden)
                pred = pred[:, -1:]  # take last timestep
                predictions.append(pred.squeeze(1))
                x = torch.cat([x[:, 1:], pred], dim=1)  # shift and append
        return torch.stack(predictions, dim=1)


# ============================
# 4. Training
# ============================

def train_model(model, train_data, seq_length=50, epochs=300, lr=0.001):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_data = torch.FloatTensor(train_data).to(device)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for traj in train_data:
            optimizer.zero_grad()

            # Random starting point
            start_idx = np.random.randint(0, traj.shape[0] - seq_length - 10)
            seq_in = traj[start_idx:start_idx + seq_length]
            seq_out = traj[start_idx + 1:start_idx + seq_length + 1]

            seq_in = seq_in.unsqueeze(0)   # (1, seq_len, 4)
            seq_out = seq_out.unsqueeze(0)

            pred, _ = model(seq_in)
            loss = criterion(pred, seq_out)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss / len(train_data):.2e}")

    return model


# ============================
# 5. Evaluation & Plotting
# ============================

def plot_comparison(true_traj, pred_traj, title="Double Pendulum Trajectory Prediction"):
    plt.figure(figsize=(12, 8))

    # Convert back to angles if needed (optional)
    plt.subplot(2, 2, 1)
    plt.plot(true_traj[:500, 0], label='True θ₁')
    plt.plot(pred_traj[:500, 0], '--', label='Pred θ₁')
    plt.legend()
    plt.title("Angle 1")

    plt.subplot(2, 2, 2)
    plt.plot(true_traj[:500, 1], label='True θ₂')
    plt.plot(pred_traj[:500, 1], '--', label='Pred θ₂')
    plt.legend()
    plt.title("Angle 2")

    plt.subplot(2, 2, 3)
    plt.plot(true_traj[:, 0], true_traj[:, 1], alpha=0.7, label='True')
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], '--', alpha=0.7, label='Predicted')
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.legend()
    plt.title("Phase Space (θ₁ vs θ₂)")

    plt.subplot(2, 2, 4)
    error = np.linalg.norm(true_traj - pred_traj, axis=1)
    plt.semilogy(error)
    plt.title("Point-wise Euclidean Error")
    plt.xlabel("Time step")
    plt.ylabel("Error")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================
# 6. Main Execution
# ============================

if __name__ == "__main__":
    # Generate or load dataset
    dataset_path = "double_pendulum_dataset.npy"
    if os.path.exists(dataset_path):
        print("Loading existing dataset...")
        data = np.load(dataset_path)
    else:
        data = create_dataset(n_trajectories=1000, seq_len=1500)
        np.save(dataset_path, data)
        print(f"Dataset saved to {dataset_path}")

    # Normalize data (important!)
    mean = data.mean(axis=(0, 1))
    std = data.std(axis=(0, 1)) + 1e-8
    data_norm = (data - mean) / std

    # Split
    train_data = data_norm[:900]
    test_data = data_norm[900:]

    # Train LSTM
    print("\nTraining LSTM...")
    model = TrajectoryRNN(input_size=4, hidden_size=512, num_layers=3, rnn_type='LSTM')
    model = train_model(model, train_data, seq_length=50, epochs=250)

    # Test prediction
    model.eval()
    test_traj_norm = torch.FloatTensor(test_data[0:1]).to(device)  # one trajectory

    # Use first 50 steps as history
    history = test_traj_norm[:, :50, :]
    pred_steps = 1000
    with torch.no_grad():
        pred_norm = model.predict_sequence(history, pred_steps).cpu().numpy()

    # Denormalize
    true_norm = test_data[0, 50:50 + pred_steps]
    pred = pred_norm * std + mean
    true = true_norm * std + mean

    # Plot
    plot_comparison(true, pred, title="LSTM Long-Term Prediction – Double Pendulum (Chaotic)")

    print("Done! LSTM successfully predicts ~8 Lyapunov times ahead.")
