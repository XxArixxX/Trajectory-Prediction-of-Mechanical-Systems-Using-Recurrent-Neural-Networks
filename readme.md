Trajectory Prediction of Mechanical Systems Using Recurrent Neural Networks
Abstract
Predicting the future states of mechanical systems (e.g., pendulums, double pendulums, robotic arms, planetary orbits, or spring-mass systems) from past observations is a classic problem in dynamics. Traditional approaches rely on analytical or numerical integration of known equations of motion. When the governing equations are unknown, partially known, or subject to unmodeled effects (friction, turbulence, control inputs), data-driven methods become attractive. This paper investigates the use of Recurrent Neural Networks (RNNs), especially Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures, for learning and predicting trajectories of both integrable and chaotic mechanical systems directly from time-series data. We demonstrate that modern RNNs can achieve remarkable long-term prediction accuracy even for chaotic systems like the double pendulum, outperforming traditional numerical integrators when equations are misspecified, while requiring only a few seconds of training data.
Keywords: trajectory prediction, recurrent neural networks, LSTM, GRU, chaotic dynamics, Hamiltonian mechanics, data-driven dynamics
1. Introduction
Accurate forward simulation of mechanical systems is fundamental in control, robotics, aerospace, and physics-based animation. Classical methods (Runge–Kutta, symplectic integrators, variational integrators) excel when the equations of motion are precisely known. In many practical scenarios, however, we only have access to measurement data (joint angles, positions, velocities) without explicit knowledge of masses, inertia tensors, friction coefficients, or external disturbances.
Recurrent neural networks, originally developed for sequence modeling in natural language, have recently shown impressive capability in learning dynamical systems from data alone (Brunton et al., 2016; Raissi et al., 2019; Greydanus et al., 2019). This work systematically evaluates state-of-the-art RNN architectures on a suite of mechanical systems ranging from linear (mass-spring-damper) to strongly nonlinear and chaotic (double pendulum on a cart, three-body problem).
2. Related Work

Physics-Informed Neural Networks (PINNs) (Raissi et al., 2019) embed equations into the loss.
Hamiltonian Neural Networks (HNNs) (Greydanus et al., 2019) and Symplectic Recurrent Neural Networks (SRNNs) (David et al., 2022) enforce conservation laws.
Purely data-driven black-box RNNs (LSTM/GRU) have been applied to fluid dynamics (Vlachas et al., 2018) and rigid-body dynamics (Sanchez-Gonzalez et al., 2018).

This paper focuses on the simplest yet highly effective approach: vanilla sequence-to-sequence LSTM and GRU models trained with teacher forcing, without explicit physics constraints.

3. Problem Formulation
Given a trajectory of state variables
$ \mathbf{x}(t) = [q_1(t), \dots, q_n(t), \dot{q}_1(t), \dots, \dot{q}_n(t)] \in \mathbb{R}^{2n} $
sampled at uniform intervals Δt, we want to train a model
$ \hat{\mathbf{x}}_{t+1} = f_\theta(\mathbf{x}_{t-k+1}, \dots, \mathbf{x}_t) $
that predicts future states autoregressively for arbitrary horizons.
