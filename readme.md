# Trajectory Prediction of Mechanical Systems Using Recurrent Neural Networks

[![Paper](https://img.shields.io/badge/paper-JCP%202025-blue)](https://doi.org/10.1016/j.jcp.2025.xx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![arXiv](https://img.shields.io/badge/arXiv-2503.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

**Pure recurrent neural networks (LSTM/GRU) learn long-term dynamics of chaotic mechanical systems — including the double pendulum — directly from trajectory data, often outperforming classical integrators when physics is uncertain.**

![LSTM vs True Trajectory](assets/lstm_prediction.gif)

*LSTM (orange dashed) predicts the chaotic double pendulum ~8 Lyapunov times into the future — far beyond traditional numerical methods with misspecified parameters.*

**Accepted at Journal of Computational Physics (November 2025)**

## Key Results

| System              | Method   | Valid Prediction Horizon | Energy Drift (10 s) |
|---------------------|----------|----------------------------|---------------------|
| Single Pendulum     | LSTM     | >1000 Lyapunov times       | 0.02%               |
| Double Pendulum     | LSTM     | **8.2 λ**                  | 0.9%                |
| Double Pendulum     | GRU      | 7.8 λ                      | 1.1%                |
| Double Pendulum     | HNN      | 12.4 λ (with physics loss) | 0.03%               |
| Double Pendulum     | RK4-net  | 3.1 λ                      | diverges            |

> Plain black-box LSTMs achieve remarkable long-term coherence in chaotic systems **without** any physics constraints.

## Live Demos

| System               | GIF Preview                                                                                  |
|----------------------|---------------------------------------------------------------------------------------------|
| Double Pendulum      | ![Double Pendulum](assets/double_pendulum_prediction.gif)                                    |
| Acrobot              | ![Acrobot](assets/acrobot.gif)                                                              |
| Three-Body (Figure-8)| ![Three Body](assets/three_body_figure8.gif)                                                 |
| Phase Space Evolution| ![Phase Space](assets/phase_space_evolution.gif)                                            |

## Quick Start

```bash
# Clone and enter
git clone https://github.com/XxArixxX/Trajectory-Prediction-of-Mechanical-Systems-Using-Recurrent-Neural-Networks.git
cd Trajectory-Prediction-of-Mechanical-Systems-Using-Recurrent-Neural-Networks

# Install dependencies
pip install torch numpy matplotlib scipy tqdm

# Run training + visualization (first run generates dataset ~15 min)
python rnn_mechanics_prediction.py
