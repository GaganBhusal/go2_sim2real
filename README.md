# Unitree Go2 Sim-to-Real Locomotion

**Reinforcement Learning Velocity-Command Locomotion for the Unitree Go2 Quadruped Robot**  


 Trained end-to-end in the [Genesis](https://genesis-world.org/) physics engine with PPO via [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl), and deployed **zero-shot** onto physical hardware via low-level DDS communication.

<p align="center">
  <a href="https://github.com/Genesis-Embodied-AI/Genesis">
    <img src="https://img.shields.io/badge/Simulator-Genesis-5B5EA6?style=for-the-badge&logo=github&logoColor=white" alt="Genesis Physics Engine" />
  </a>
  <a href="https://github.com/leggedrobotics/rsl_rl">
    <img src="https://img.shields.io/badge/RL%20Framework-rsl__rl-2E7D32?style=for-the-badge&logo=github&logoColor=white" alt="rsl_rl" />
  </a>
  <a href="https://github.com/unitreerobotics/unitree_sdk2_python">
    <img src="https://img.shields.io/badge/Hardware-Unitree%20Robotics-E53935?style=for-the-badge&logo=github&logoColor=white" alt="Unitree Robotics" />
  </a>
  <a href="https://www.ran.org.np/">
    <img src="https://img.shields.io/badge/Community-RAN-0052CC?style=for-the-badge" alt="Robotics Association of Nepal" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License" />
  </a>
</p>

<!-- --- -->

<!-- DEMO VIDEOS -->
<div align="center">

### 🎬 Real Hardware Deployment Demos

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/shorts/0ZMWlH1WIyI" target="_blank">
        <img src="https://img.youtube.com/vi/0ZMWlH1WIyI/maxresdefault.jpg" alt="Go2 Hardware Locomotion Demo 1" width="100%" />
      </a>
      <br />
      <a href="https://www.youtube.com/shorts/0ZMWlH1WIyI"><b>▶️ Watch Demo 1 (YouTube Shorts)</b></a>
    </td>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/shorts/KFbTj_H0R_Y" target="_blank">
        <img src="https://img.youtube.com/vi/KFbTj_H0R_Y/maxresdefault.jpg" alt="Go2 Hardware Locomotion Demo 2" width="100%" />
      </a>
      <br />
      <a href="https://www.youtube.com/shorts/KFbTj_H0R_Y"><b>▶️ Watch Demo 2 (YouTube Shorts)</b></a>
    </td>
  </tr>
</table>

</div>

---

## Table of Contents
- [Overview](#overview)
- [Method & Architecture](#method--architecture)
  - [Control Hierarchy & Rates](#control-hierarchy--rates)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Domain Randomization](#domain-randomization)
  - [Reward Formulation](#reward-formulation)
  - [PD Gain Selection & Actuator Dynamics](#pd-gain-selection--actuator-dynamics)
  - [Sim-to-Real Transfer Pipeline](#sim-to-real-transfer-pipeline)
- [Hardware Experiments](#hardware-experiments)
  - [Actuator Gain Sweep](#actuator-gain-sweep)
  - [Failure Modes & Safety Cutoffs](#failure-modes--safety-cutoffs)
- [Limitations](#limitations)
- [Repo Structure](#repo-structure)
- [Setup and Usage](#setup-and-usage)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Simulation Training](#2-simulation-training)
  - [3. Simulation Playback & Interactive Control](#3-simulation-playback--interactive-control)
  - [4. Real Hardware Deployment](#4-real-hardware-deployment)
- [Acknowledgements & References](#acknowledgements--references)
- [License](#license)

---

## Overview

This repository presents an end-to-end reinforcement learning framework for dynamic, blind velocity-tracking locomotion on the **Unitree Go2** quadruped robot, transferring **zero-shot** from simulation to physical hardware.

```
       SIMULATION (Genesis GPU)                      LEARNING (rsl_rl)                     HARDWARE (Unitree Go2)
 ┌───────────────────────────────────┐    ┌───────────────────────────────────┐    ┌───────────────────────────────────┐
 │ • 4,096 Vectorized Environments   │    │ • Asymmetric Actor-Critic (PPO)   │    │ • Zero-Shot Hardware Transfer     │
 │ • 500 Hz Physics (10 Substeps)    │───►│ • Online Observation Normalizer   │───►│ • Low-Level CycloneDDS (LowCmd)   │
 │ • Rigorous Domain Randomization   │    │ • Cyclical Trot Clock Generator   │    │ • Kp=40.0, Kd=4.9 (Damped QDD)    │
 │ • Mass, Friction, Gain Sweeps     │    │ • 22 Balanced Reward Objectives   │    │ • Auto Stand-Up & E-Stop Safety   │
 └───────────────────────────────────┘    └───────────────────────────────────┘    └───────────────────────────────────┘
```

### Key Highlights
- **High-Throughput Simulation in Genesis:** Vectorized GPU rigid-body dynamics simulating 4,096 parallel environments with 500 Hz physics substeps ($dt = 0.002\,\text{s}$, 10 substeps per control step), closely matching real actuator integration.
- **Asymmetric Actor-Critic with PPO:** Trained with [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) using an asymmetric observation scheme: a 47-dimensional proprioceptive actor policy for onboard deployment and a 55-dimensional privileged critic network leveraging ground-truth physics states.
- **Aggressive Domain Randomization:** Bridges the sim-to-real gap across base mass $[-1.0, +2.5]\,\text{kg}$, CoM offset $\pm 4\,\text{cm}$, friction $[0.05, 3.0]$, actuator gains ($K_p \in [20, 80]$, $K_d \in [0.2, 5.0]$), joint damping, dry friction loss, rotor armature, motor torque scaling, external pushes up to $150\,\text{N}$, and sensor noise.
- **Actuator Dynamics & Zero-Shot Transfer:** Quasi-direct drive (QDD) actuators with low gear reduction ($\sim 6.33:1$) transmit high touchdown shock waves back into the motors. Deploying with tuned electronic damping ($K_p = 40.0, K_d = 4.9$) eliminates contact chatter without needing laggy action EMA filters that degrade closed-loop stability.
- **Embedded Hardware Safety:** Direct communication via Unitree's low-level DDS API (`rt/lowcmd` and `rt/lowstate`) at 500 Hz. Features automated 2-second stand-up interpolation, explicit kinematic remapping (`SIM2REAL` / `REAL2SIM`), and real-time gravity projection fall detection ($g_z > -0.5 \implies \text{tilt} > 60^\circ$) with safe soft-damping shutdown.

---

<!-- ## Results

Locomotion performance metrics evaluated on physical hardware from onboard high-frequency telemetry logs (`robot_data_control.csv` at 500 Hz, `robot_data_policy.csv` at 50 Hz, totaling 25,842 control steps over 51.7 seconds of continuous walking):

| Metric | Simulation (Genesis) | Real Hardware (Go2) | Units |
|---|:---:|:---:|:---:|
| **Continuous Walk Duration (No Falls / E-Stops)** | $> 1000$ steps | **51.70** (25,842 steps) | $\text{s}$ |
| **Commanded Speed Envelope ($v_x$)** | $[-1.00,\, 1.00]$ | **$[-0.89,\, 0.94]$** (mean $0.13$) | $\text{m/s}$ |
| **Commanded Lateral Speed Envelope ($v_y$)** | $[-1.00,\, 1.00]$ | **$[-0.73,\, 0.78]$** (mean $-0.00$) | $\text{m/s}$ |
| **Commanded Turn Rate Envelope ($\omega_z$)** | $[-1.00,\, 1.00]$ | **$[-0.65,\, 0.95]$** (mean $0.17$) | $\text{rad/s}$ |
| **Yaw-Rate Tracking RMSE ($\omega_z$)** | $\approx 0.08$ | **0.154** ($8.80^\circ/\text{s}$) | $\text{rad/s}$ |
| **Yaw-Rate Tracking MAE ($\omega_z$)** | $\approx 0.05$ | **0.113** ($6.49^\circ/\text{s}$) | $\text{rad/s}$ |
| **Action First-Difference ($\Vert \Delta a_t \Vert$)** | $\approx 1.20$ | **1.597** (RMS $1.789$) | $\text{rad}$ |
| **Action Second-Difference ($\Vert \Delta^2 a_t \Vert$)** | $\approx 1.80$ | **2.396** (RMS $2.769$) | $\text{rad}$ |
| **Mean Action Rate ($\Vert \dot{a} \Vert$)** | $\approx 60.0$ | **80.12** | $\text{rad/s}$ |
| **Mean Action Acceleration ($\Vert \ddot{a} \Vert$)** | $\approx 4500.0$ | **6026.74** | $\text{rad/s}^2$ |
| **Mean Body Tilt Angle** | $\approx 1.5^\circ$ | **2.26** (RMS $2.45^\circ$) | $\text{deg}$ |
| **Peak Body Tilt Angle** | $< 5.0^\circ$ | **7.46** (Roll $[-5.1^\circ, 3.9^\circ]$, Pitch $[-6.0^\circ, 2.5^\circ]$) | $\text{deg}$ |
| **Joint Position Tracking RMSE ($q_{\text{des}} - q_{\text{meas}}$)** | $\approx 0.12$ | **0.232** ($13.27^\circ$) | $\text{rad}$ |
| **Mean Absolute Joint Torque** | $\approx 3.8$ | **4.24** | $\text{N}\cdot\text{m}$ |
| **Overall RMS Joint Torque** | $\approx 5.5$ | **6.95** | $\text{N}\cdot\text{m}$ |
| **Peak Joint Torque (Max across motors)** | $35.55$ (limit) | **37.27** (calves) | $\text{N}\cdot\text{m}$ |
| **Mean Electrical Power Consumption** | — | **172.57** (std $104.08$) | $\text{W}$ |
| **Peak Electrical Power** | — | **1102.38** (during sprint bursts) | $\text{W}$ |
| **Total Energy Consumed** | — | **8.93** ($2.48\,\text{Wh}$ over $51.7\,\text{s}$) | $\text{kJ}$ |
| **Mean Foot Ground Contact Force** | $\approx 38.0$ | **40.0** (FR: 43.8, FL: 41.1, RR: 38.2, RL: 36.8) | $\text{N}$ |
| **Peak Foot Contact Force** | $\approx 150.0$ | **191.0** (FL) | $\text{N}$ |
| **Average Motor Core Temperature** | — | **34.3** (range $[32.0,\, 37.0]$) | $^\circ\text{C}$ |

> **Telemetry & Measurement Notes:**
> - **Yaw-Rate Tracking:** Evaluated directly by comparing high-level commanded yaw rate $\omega_z^{\text{cmd}}$ against onboard IMU gyroscope readings ($\omega_z^{\text{meas}}$) downsampled to the 50 Hz policy horizon. Tracking error remains tight ($\text{RMSE} = 8.80^\circ/\text{s}$), with transient spikes occurring only during high-acceleration stick reversals.
> - **Action Smoothness:** On hardware, high-frequency impact shocks subtly propagate into joint velocity feedbacks $\dot{q}$, leading to a slightly higher action acceleration ($\Vert \ddot{a} \Vert = 6026.7\,\text{rad/s}^2$) compared to idealized simulation contacts.
> - **Power & Torque:** Calves absorb the majority of ground impacts, operating at $8.8\text{--}11.7\,\text{N}\cdot\text{m}$ RMS torque (vs. $2.7\text{--}3.4\,\text{N}\cdot\text{m}$ on hips and $5.3\text{--}5.9\,\text{N}\cdot\text{m}$ on thighs). Average power draw of $172.6\,\text{W}$ at $\approx 30.5\,\text{V}$ confirms energy-efficient cruising trot dynamics.

--- -->

## Method & Architecture

### Control Hierarchy & Rates

The control architecture is split across two distinct update frequencies:
- **High-Level Policy Loop (50 Hz, $dt = 0.02\,\text{s}$):** An MLP actor policy evaluates the 47-dimensional normalized observation vector and outputs 12-dimensional joint position target deltas.
- **Low-Level Motor PD Controller (500 Hz, $dt = 0.002\,\text{s}$):** Motors track the commanded position targets at 500 Hz using low-level onboard PD control:
  $$\tau_i = K_p (q_{i,\text{des}} - q_{i,\text{meas}}) + K_d (\dot{q}_{i,\text{des}} - \dot{q}_{i,\text{meas}})$$
  In simulation, Genesis executes 10 physics substeps per policy tick ($10 \times 0.002\,\text{s} = 0.02\,\text{s}$).

<!-- ```
[Sensors: IMU + Motor Encoders]
               │ (500 Hz)
               ▼
   [State Estimation / Remap]
               │ (50 Hz downsampled)
               ▼
    [Actor MLP Policy (47-dim)]  ◄─── [User Velocity Commands]
               │
               ▼ (Target Position Deltas)
   [Kinematic Remap & Safety Bounds]
               │ (500 Hz low-level loop)
               ▼
    [Go2 Actuators (LowCmd DDS)]  ───► Kp = 40.0, Kd = 4.9
``` -->

---

### Observation Space

The environment uses an asymmetric actor-critic architecture. The policy actor network receives proprioceptive observations (47-dim), while the critic has access to privileged simulation states (55-dim).

#### 1. Policy Actor Observation Space (47 dimensions)

| Indices | Dim | Feature Name | Scale Factor | Mathematical Expression / Source | Physical Meaning |
|:---:|:---:|---|:---:|---|---|
| `[0:3]` | 3 | Base Angular Velocity | $\mathbf{0.50}$ | $\omega_{\text{gyro}} \times 0.5$ | Filtered 3D angular velocity in robot base frame ($\text{rad/s}$) |
| `[3:6]` | 3 | Projected Gravity Vector | $\mathbf{1.00}$ | $[-g_x, -g_y, -g_z] = \mathbf{R}(q_{\text{base}})^T \begin{bmatrix}0\\0\\-1\end{bmatrix}$ | Direction of gravity vector expressed in robot base frame |
| `[6:18]` | 12 | Joint Position Error | $\mathbf{1.00}$ | $q - q_{\text{default}}$ | Joint angles offset from nominal standing posture ($\text{rad}$) |
| `[18:30]` | 12 | Joint Velocities | $\mathbf{0.05}$ | $\dot{q} \times 0.05$ | Scaled joint angular velocities from motor encoders ($\text{rad/s}$) |
| `[30:42]` | 12 | Previous Actions | $\mathbf{1.00}$ | $a_{t-1}$ | Target joint delta commands from previous policy cycle |
| `[42:45]` | 3 | Commanded Velocity Targets | $\mathbf{1.00}$ | $[v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_z^{\text{cmd}}]$ | User velocity commands in body frame ($\text{m/s}, \text{rad/s}$) |
| `[45:47]` | 2 | Periodic Gait Clock | $\mathbf{1.00}$ | $[\sin(2\pi \phi),\, \cos(2\pi \phi)]$ | Cyclic phase clock ($\phi = (t \bmod T)/T$, period $T = 0.5\,\text{s}$) |

#### 2. Critic Privileged Observation Space (55 dimensions)

During training in Genesis, the critic value network evaluates state values using ground-truth physical quantities, allowing asymmetric learning without relying on noisy state estimation:

| Indices | Dim | Feature Name | Scale Factor | Mathematical Expression / Source | Physical Meaning |
|:---:|:---:|---|:---:|---|---|
| `[0:3]` | 3 | Base Linear Velocity (Body Frame) | $\mathbf{2.00}$ | $\mathbf{v}_{\text{body}} \times 2.0 = \mathbf{R}(q_{\text{base}})^T \mathbf{v}_{\text{world}} \times 2.0$ | Ground-truth 3D torso linear velocity ($v_x, v_y, v_z$) without sensor lag |
| `[3:6]` | 3 | Base Angular Velocity | $\mathbf{0.50}$ | $\omega_{\text{base}} \times 0.5$ | Ground-truth 3D torso angular velocity ($\omega_x, \omega_y, \omega_z$) in base frame |
| `[6:7]` | 1 | Base Torso Height ($z$) | $\mathbf{1.00}$ | $z_{\text{base}} \times 1.0$ | Ground-truth vertical distance of robot base frame above terrain plane ($\text{m}$) |
| `[7:10]` | 3 | Projected Gravity Vector | $\mathbf{1.00}$ | $[-g_x, -g_y, -g_z] = \mathbf{R}(q_{\text{base}})^T \begin{bmatrix}0\\0\\-1\end{bmatrix}$ | Exact gravitational direction vector without IMU integration drift |
| `[10:22]` | 12 | Joint Position Errors | $\mathbf{1.00}$ | $q - q_{\text{default}}$ | Exact joint angles offset from default standing posture ($\text{rad}$) |
| `[22:34]` | 12 | Joint Angular Velocities | $\mathbf{0.05}$ | $\dot{q} \times 0.05$ | Clean joint angular velocities without encoder differentiation noise ($\text{rad/s}$) |
| `[34:37]` | 3 | Commanded Velocity Targets | $\mathbf{1.00}$ | $[v_x^{\text{cmd}}, v_y^{\text{cmd}}, \omega_z^{\text{cmd}}]$ | Commanded velocity setpoints in robot body frame ($\text{m/s}, \text{rad/s}$) |
| `[37:41]` | 4 | Feet Flight Air Time | $\mathbf{1.00}$ | $\mathbf{t}_{\text{air}} = [t_{\text{air}}^{\text{FL}}, t_{\text{air}}^{\text{FR}}, t_{\text{air}}^{\text{RL}}, t_{\text{air}}^{\text{RR}}]$ | Accumulated swing phase flight duration for each individual leg ($\text{s}$) |
| `[41:53]` | 12 | Previous Actions | $\mathbf{1.00}$ | $a_{t-1}$ | Preceding policy joint position target deltas emitted by the actor |
| `[53:54]` | 1 | Periodic Gait Clock (Sin) | $\mathbf{1.00}$ | $\sin(2\pi \phi),\;\phi = (t \bmod T)/T$ | Fundamental sine harmonic of cyclical trotting gait generator ($T = 0.5\,\text{s}$) |
| `[54:55]` | 1 | Periodic Gait Clock (Cos) | $\mathbf{1.00}$ | $\cos(2\pi \phi),\;\phi = (t \bmod T)/T$ | Fundamental cosine harmonic of cyclical trotting gait generator ($T = 0.5\,\text{s}$) |

---

### Action Space

The policy outputs 12-dimensional joint position target deltas $a_t \in \mathbb{R}^{12}$:
$$q_{\text{target}} = q_{\text{default}} + a_t \times \text{action\_scale}$$
where:
- $\text{action\_scale} = 0.25$
- Nominal default joint posture $q_{\text{default}}$ (hip, thigh, calf):
  - Hip: $0.0^\circ$ ($0.0\,\text{rad}$)
  - Thigh: $+50.0^\circ$ ($0.8726\,\text{rad}$) for FL/FR, $+55.0^\circ$ ($0.9599\,\text{rad}$) for RL/RR
  - Calf: $-90.0^\circ$ ($-1.5708\,\text{rad}$) for all legs
- Clamped within mechanical joint safety bounds:
  - Hip abduction/adduction: $[-20^\circ, +20^\circ]$
  - Thigh flexion/extension: $[-90^\circ, +200^\circ]$ (front), $[-30^\circ, +260^\circ]$ (rear)
  - Calf flexion/extension: $[-156^\circ, -48^\circ]$

---

### Domain Randomization

To bridge the sim-to-real reality gap, aggressive domain randomization is applied at environment resets and runtime simulation steps:

| Parameter Name | Config Variable | Nominal Value | Randomized Range | Unit | Perturbation Formulation | Physical Meaning & Sim-to-Real Role |
|---|---|:---:|:---:|:---:|---|---|
| **Base Added Mass** | `mass_shift_range` | $0.0$ | $[-1.00,\, +2.50]$ | $\text{kg}$ | $m_{\text{base}} \leftarrow m_{\text{base}} + \Delta m$ | Accounts for battery aging, payloads, and compute hardware additions |
| **Center of Mass (CoM) Displacement** | `com_shift_range` | $(0, 0, 0)$ | $[\pm 0.04,\, \pm 0.04,\, \pm 0.04]$ | $\text{m}$ | $\mathbf{r}_{\text{CoM}} \leftarrow \mathbf{r}_{\text{CoM}} + \Delta \mathbf{r}$ | Robustness against asymmetric weight distribution causing pitch/roll tilt |
| **Ground Friction Coefficient** | `friction_range` | $1.00$ | $[0.05,\, 3.00]$ | — | $\mu \sim \mathcal{U}(0.05, 3.0)$ | Generalizes across slippery lab tiles, rugs, wood, and rough asphalt |
| **Joint Viscous Damping** | `damping_range` | $0.05$ | $[0.02,\, 0.40]$ | $\text{N}\cdot\text{s}/\text{rad}$ | $D_j \sim \mathcal{U}(0.02, 0.40)$ | Emulates motor back-EMF variations and bearing grease viscosity |
| **Joint Dry Friction Loss** | `frictionloss_range` | $0.05$ | $[0.02,\, 0.50]$ | $\text{N}\cdot\text{m}$ | $\tau_{\text{loss}} \sim \mathcal{U}(0.02, 0.50)$ | Bridges Coulomb friction and planetary gear stiction across joints |
| **Rotor Armature Inertia** | `armature_range` | $0.002$ | $[0.001,\, 0.008]$ | $\text{kg}\cdot\text{m}^2$ | $I_{\text{rotor}} \sim \mathcal{U}(0.001, 0.008)$ | Models motor rotor inertia reflected through the 6.33:1 gearbox reduction |
| **Joint Proportional Gain ($K_p$)** | `kp_range` | $40.0$ | $[20.0,\, 80.0]$ | $\text{N}\cdot\text{m}/\text{rad}$ | $K_p \sim \mathcal{U}(20.0, 80.0)$ | Invariance to joint position tracking stiffness; deployed at $K_p = 40.0$ |
| **Joint Derivative Gain ($K_d$)** | `kv_range` | $1.0$ | $[0.20,\, 5.00]$ | $\text{N}\cdot\text{m}\cdot\text{s}/\text{rad}$ | $K_d \sim \mathcal{U}(0.20, 5.00)$ | Allows high electronic damping ($K_d = 4.9$) to quench touchdown vibration |
| **Motor Torque Limit Scaling** | `torque_limit_scale_range` | $1.00$ | $[0.70,\, 1.00]$ | — | $\tau_{\text{max}} \leftarrow \alpha \cdot \tau_{\text{nominal}}$ | Prevents over-reliance on peak torque; accommodates voltage drops |
| **External Push Impulses** | `max_push_force` | $0.0$ | $[0.0,\, 150.0]$ ($p = 0.01$) | $\text{N}$ | $\mathbf{F}_{\text{push}} \sim \mathcal{U}(0, 150)\,\text{N}$ | Teaches balance recovery and active foot placement under external collisions |
| **Gyroscope Observation Noise** | `actor_ang_vel` | $0.0$ | $\sigma = 0.05$ | $\text{rad/s}$ | $\omega_{\text{gyro}} + \mathcal{N}(0, 0.05^2)$ | Emulates real MEMS inertial sensor bias drift and high-frequency noise |
| **Gravity Vector Noise** | `actor_gravity` | $0.0$ | $\sigma = 0.02$ | — | $\mathbf{g}_{\text{proj}} + \mathcal{N}(0, 0.02^2)$ | Accounts for orientation filter estimation uncertainty during dynamic motion |
| **Joint Encoder Position Noise** | `actor_dof_pos` | $0.0$ | $\sigma = 0.01$ ($0.57^\circ$) | $\text{rad}$ | $q + \mathcal{N}(0, 0.01^2)$ | Models optical/magnetic encoder resolution limits and link compliance |
| **Joint Velocity Noise** | `actor_dof_vel` | $0.0$ | $\sigma = 0.50$ | $\text{rad/s}$ | $\dot{q} + \mathcal{N}(0, 0.50^2)$ | Models discrete numerical differentiation noise from hardware motor state |

---

### Reward Formulation

The total reinforcement learning reward is composed of 22 individual objectives covering command tracking, gait shaping, energy minimization, and safety constraints. 

Per policy step, the aggregated reward is scaled by the policy timestep ($dt = 0.02\,\text{s}$):
$$R_{\text{step}} = dt \times \left( \sum_{i} w_i \cdot r_i + r_{\text{termination}} \right)$$

| Category | Reward Term | Config Parameter | Scale ($w_i$) | Mathematical Expression / Condition | Description & Behavioral Impact |
|---|---|---|:---:|---|---|
| **Command Tracking** | Linear Velocity X | `reward_scales_tracking_vx` | $\mathbf{+1.00}$ | $\exp\left(-\frac{(v_x^{\text{cmd}} - v_x)^2}{\sigma_{vx}}\right),\;\sigma_{vx}=0.25$ | Accurately track commanded forward / backward body velocity |
| | Linear Velocity Y | `reward_scales_tracking_vy` | $\mathbf{+1.00}$ | $\exp\left(-\frac{(v_y^{\text{cmd}} - v_y)^2}{\sigma_{vx}}\right),\;\sigma_{vx}=0.25$ | Accurately track commanded lateral strafing velocity |
| | Yaw Rate Z | `reward_scales_tracking_wz` | $\mathbf{+1.00}$ | $\exp\left(-\frac{(\omega_z^{\text{cmd}} - \omega_z)^2}{\sigma_{wz}}\right),\;\sigma_{wz}=0.25$ | Accurately track commanded rotational heading turn rate |
| **Gait & Cadence** | Feet Air Time | `reward_scales_feet_air_time` | $\mathbf{+1.50}$ | $\sum_{i=1}^4 (t_{\text{air}, i} - 0.15)\cdot \mathbb{I}_{\text{first\_contact}}$ | Encourages dynamic flight phase ($150\,\text{ms}$ target) and prevents leg dragging |
| | Phase Contact Sync | `reward_scales_contact` | $\mathbf{+0.50}$ | $\frac{1}{4}\sum \neg(\text{contact}_i \oplus \text{stance}_i)$ | Synchronizes foot stance with cyclic clock reference ($\phi_{\text{stance}} < 0.55$) |
| | Feet Swing Height | `reward_scales_feet_swing_height` | $\mathbf{+0.50}$ | $\sum \exp\left(-\frac{(h_i - h_{\text{target}}(\phi))^2}{0.005}\right) \cdot \mathbb{I}_{\text{swing}}$ | Tracks dynamic sinusoidal clearance arc ($10\,\text{cm}$ apex) during swing |
| | Foot Slip Penalty | `reward_scales_foot_slip` | $\mathbf{-0.25}$ | $\sum_{i=1}^4 \Vert v_{xy, i}^{\text{foot}} \Vert^2 \cdot \mathbb{I}_{\text{contact}, i}$ | Penalizes foot skidding while in contact with the ground |
| | Foot Clearance | `reward_scales_foot_clearance` | $\mathbf{0.00}$ | $\sum \mathbb{I}_{\text{swing}} \cdot (h_{\text{target}} - h_z)^2 \cdot \Vert v_{xy} \Vert$ | Secondary obstacle clearance term (disabled in favor of swing height) |
| **Base Posture** | Base Height | `reward_scales_base_height` | $\mathbf{-30.00}$ | $(h_{\text{base}} - 0.31)^2$ | Enforces nominal walking torso clearance ($0.31\,\text{m}$ above ground) |
| | Vertical Velocity | `reward_scales_lin_vel_z` | $\mathbf{-2.00}$ | $v_z^2$ | Suppresses vertical body bobbing and energy-wasting torso bounce |
| | Base Angular Rates | `reward_scales_base_roll_pitch_rates` | $\mathbf{-0.05}$ | $\omega_{\text{roll}}^2 + \omega_{\text{pitch}}^2$ | Minimizes parasite body rocking in roll and pitch orientations |
| | Stand Still Posture | `reward_scales_stand_still` | $\mathbf{-0.40}$ | $\sum \vert q_j - q_{\text{default}, j} \vert \cdot \mathbb{I}_{\Vert v^{\text{cmd}} \Vert < 0.1}$ | Holds clean upright posture without wandering when zero command is issued |
| | Nominal Joint Posture | `reward_scales_dof_pos` | $\mathbf{-0.50}$ | $\sum_{j=1}^{12} (q_j - q_{\text{default}, j})^2$ | Regularizes all joints toward default standing bias angles |
| | Hip Splay Deviation | `reward_scales_hip_deviation_penalty` | $\mathbf{-2.00}$ | $\sum_{j \in \text{hips}} q_j^2$ | Restricts lateral hip abduction/adduction to prevent wide-splay walking |
| **Smoothness & Energy** | Action Rate ($\Delta a$) | `reward_scales_action_rate` | $\mathbf{-0.01}$ | $\sum_{j=1}^{12} (a_{t, j} - a_{t-1, j})^2$ | Penalizes high-frequency command jitter and motor heating |
| | Joint Accelerations | `reward_scales_dof_acc` | $\mathbf{-5.0\times 10^{-7}}$ | $\sum_{j=1}^{12} \left(\frac{\dot{q}_{t, j} - \dot{q}_{t-1, j}}{dt}\right)^2$ | Suppresses mechanical shock waves and jerky actuator acceleration |
| | Joint Velocities | `reward_scales_dof_vel` | $\mathbf{-1.0\times 10^{-3}}$ | $\sum_{j=1}^{12} \dot{q}_j^2$ | Encourages energy efficiency by penalizing excessive motor spinning speed |
| | Motor Torques | `reward_scales_torques` | $\mathbf{-1.0\times 10^{-5}}$ | $\sum_{j=1}^{12} \tau_j^2$ | Minimizes $I^2 R$ electrical power dissipation and actuator thermal load |
| **Safety & Limits** | Soft Joint Limits | `reward_scales_dof_pos_limits_soft` | $\mathbf{-0.10}$ | $\sum \left( \max(0, q_{\text{min}} + \delta - q)^2 + \max(0, q - (q_{\text{max}} - \delta))^2 \right)$ | Soft barrier penalty when joint positions enter within 10% of physical limits |
| | Body Link Collision | `reward_scales_collision` | $\mathbf{-1.00}$ | $\sum \mathbb{I}_{\text{contact}}(\text{hips, thighs})$ | Penalizes self-collisions and ground contacts on structural links |
| | Survival Bonus | `reward_scales_survival` | $\mathbf{+1.00}$ | $+1.0$ per alive timestep | Constant positive incentive to prevent premature termination |
| | Terminal Fall Penalty | `reward_scales_termination_penalty` | $\mathbf{-500.00}$ | $\mathbb{I}\left(\vert\text{roll}\vert > 1.0 \lor \vert\text{pitch}\vert > 1.0 \lor h_{\text{base}} < 0.18\right)$ | Massive catastrophic penalty upon fall, rollover, or base collapse |

---

### PD Gain Selection & Actuator Dynamics

A key hardware insight discovered during deployment revolves around the **actuator damping requirements** of the Go2:

- **Optimal Hardware Operating Point:** Setting **$K_p = 40.0$** and **$K_d = 4.5 - 4.9$** provided the exact critical damping needed to absorb contact transients immediately. Because the policy was trained across $K_d \in [0.2, 5.0]$, this high-damping configuration stayed well within the learned policy distribution.
- **No Action Filtering (EMA):** While an exponential moving average filter on policy actions is sometimes used to suppress jitter, it introduces unmodeled phase lag that breaks the tight 50 Hz closed-loop balance. Tuning the physical PD damping eliminated jitter at the source without adding latency.

---

### Sim-to-Real Transfer Pipeline

#### 1. Kinematic Joint Mapping (Full Names & Ordering)

Genesis and the Unitree Low-Level SDK use fundamentally different joint serialization conventions:
- **Genesis Simulator (Type-Grouped Order):** Groups all 4 hips first (`FL, FR, RL, RR`), then all 4 thighs, then all 4 calves.
- **Unitree SDK Hardware (Leg-Sequential Order):** Addresses actuators grouped leg-by-leg (`FR -> FL -> RR -> RL`), ordering hip, thigh, calf for each leg.

To maintain real-time correspondence without copy overhead, bi-directional index permutation tensors are applied:
```python
# Real motor command target = policy_action[SIM2REAL]
SIM2REAL = [1, 5, 9,  0, 4, 8,  3, 7, 11,  2, 6, 10]

# Sim observation state = real_sensor_state[REAL2SIM]
REAL2SIM = [3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8]
```

#### 2. Two-Stage Startup Sequence

Before handing over motor control to the neural network, the deployment script executes a 2-second ($1000$ iterations at $500\,\text{Hz}$) linear position interpolation from the robot's resting configuration to $q_{\text{default}}$. Once the nominal stance is reached, the 50 Hz policy loop starts.

#### 3. Safety Fall Detection & Damping Mode

- **Tilt Limit:** Base pitch and roll are monitored via the projected gravity component:
  $$g_z = -(1.0 - 2.0 \cdot (q_x^2 + q_y^2))$$
  If $g_z > -0.5$ (equivalent to base tilt exceeding $60^\circ$), an emergency stop triggers immediately.
- **Safe Damping Cutoff:** Upon an E-stop or `Ctrl+C` interrupt, target positions are cleared and all 12 motors enter zero-stiffness damping mode ($K_p = 0.0, K_d = 2.0$) for $0.5\,\text{s}$, letting the robot collapse gently without violent torque spikes.

---

## Hardware Experiments

### Actuator Gain Sweep

During physical tuning, we tested various gain combinations on real ground surfaces:

| $K_p$ | $K_d$ | Hardware Behavior | Stability Assessment |
|:---:|:---:|---|:---:|
| `40.0` | `1.0` | High-frequency foot rattle on hard ground; audible chatter upon touchdown. | ⚠️ Unstable contact |
| `40.0` | `3.0` | Improved gait; occasional 1–2 cycle oscillation after sudden steps. | ⚠️ Marginal |
| `60.0` | `4.0` | Severe stiffness shudder; legs fought floor impacts violently. | ❌ Failed |
| `40.0` | `4.5` | Smooth locomotion; contact impacts cleanly absorbed; stable trot. | ✅ Robust |
| **`40.0`** | **`4.9`** | **Clean, deadbeat contact damping; quiet stance phase; matches sim quality.** | 🏆 **Optimal** |

---


### Failure Modes & Safety Cutoffs


1. **Extreme Slopes ($> 25^\circ$):** Lack of elevation awareness leads to early ground contact on steep inclines.
2. **Safety Fall Detection Trigger:** Verified by tipping the robot past $60^\circ$ on a harness: motor torque instantly cut into soft damping ($K_p = 0.0, K_d = 2.0$), preventing motor over-current or gearbox damage.

---

## Limitations

- **Actuator Damping Boundary:** Deployed $K_d = 4.9$ operates right near the upper limit of the domain-randomized training range ($K_d \in [0.2, 5.0]$). Expanding the training range to $K_d \in [1.0, 7.0]$ would provide greater safety margin.
- **Blind Locomotion:** The policy does not incorporate exteroceptive perception (depth cameras or LiDAR). Steps onto obstacles exceeding $8\,\text{cm}$ rely entirely on mechanical compliance and proprioceptive push-back.
- **No Ground-Truth Linear Velocity Estimator:** The observation space excludes base linear velocity ($v_x, v_y$) to prevent policy dependence on noisy state estimation; however, this creates a slight tracking lag during aggressive velocity transients.

---

## Repo Structure

```text
go2_sim2real/
├── configs/
│   ├── env_configs.py              # Unified WalkEnvCfg combining all sub-configs
│   ├── sim_config.py              # Genesis simulator settings (dt=0.02, 10 substeps)
│   ├── robot_config.py            # Go2 URDF paths, joint limits, nominal stance
│   ├── task_config.py             # Velocity ranges, reward scales, gait clock periods
│   ├── domain_randomize_config.py # DR bounds (mass, CoM, friction, Kp, Kd, pushes)
│   └── train_params.yaml          # PPO actor-critic network hyperparameters
├── src/
│   └── env/
│       ├── base_env.py            # Base vectorized quadruped environment in Genesis
│       ├── env_walk_feet_rewards.py # Walking environment with gait & foot rewards
│       └── rewards.py             # Modular reward computation functions
├── scripts/
│   ├── train.py                   # Main PPO training script using Genesis + rsl_rl
│   └── walk_controller_with_keyboard.py # Interactive keyboard sim evaluation
├── unitree_sdk2_python/
│   └── example/go2/low_level/
│       └── custom_test_file.py    # Zero-shot real robot deployment script (500Hz/50Hz)
├── deployment/
│   └── go2_walk_deployment.py     # Standalone deployment utilities
├── logs/                          # Checkpoints and TensorBoard event logs
├── requirements.txt               # Python package dependencies
└── README.md
```

---

## Setup and Usage

### 1. Environment Setup

Clone this repository and create a dedicated Conda environment:

```bash
git clone https://github.com/<your-username>/go2_sim2real.git
cd go2_sim2real

conda create -n rl_robot python=3.10 -y
conda activate rl_robot

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Genesis simulator and rsl_rl
pip install genesis-world
pip install rsl-rl

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

To enable hardware communication, install the Unitree SDK 2 Python bindings:
```bash
cd unitree_sdk2_python
pip install -e .
cd ..
```

---

### 2. Simulation Training

Train the locomotion policy in Genesis across 4,096 parallel environments with domain randomization:

```bash
# Train from scratch with domain randomization enabled
python scripts/train.py \
  --exp_name go2_walk_seven \
  --domain \
  --num_envs 4096 \
  --max_iter 10000

# Resume training from a specific checkpoint
python scripts/train.py \
  --exp_name go2_walk_seven \
  --domain \
  --resume \
  --ckpt 5000
```

Monitor training progress via TensorBoard:
```bash
tensorboard --logdir logs/
```

---

### 3. Simulation Playback & Interactive Control

Test your trained checkpoint interactively in Genesis using real-time keyboard control:

```bash
python scripts/walk_controller_with_keyboard.py \
  --exp_name go2_walk_seven \
  --ckpt 10000 \
  --device cuda
```

**Keyboard Controls:**
| Key | Action |
|---|---|
| `Up` / `Down` | Forward / Backward Velocity ($v_x \in [-1.0, 1.0]\,\text{m/s}$) |
| `Left` / `Right` | Lateral Left / Right Velocity ($v_y \in [-1.0, 1.0]\,\text{m/s}$) |
| `Numpad 1` / `Numpad 3` | Turn Left / Right Yaw Rate ($\omega_z \in [-1.0, 1.0]\,\text{rad/s}$) |

---

### 4. Real Hardware Deployment

#### ⚠️ Safety Preparation Checklist
1. **Gantry / Stand:** Suspend the robot on a hoist or place it securely on a central support stand with feet off the ground for initial verification.
2. **Emergency Stop Ready:** Ensure access to the keyboard running the script (`Ctrl+C` triggers graceful damping shutdown).
3. **Network Configuration:** Connect an Ethernet cable from your control computer to the Go2 or connect over its internal Wi-Fi. Ensure CycloneDDS is bound to the appropriate network interface.

#### Launching on the Robot

1. Put the robot in low-level mode (release sport mode):
   The deployment script automatically handles releasing motion control via `MotionSwitcherClient` and `SportClient`.

2. Execute the low-level deployment script with root permissions for DDS networking:
   ```bash
   sudo $(which python) unitree_sdk2_python/example/go2/low_level/custom_test_file.py
   ```

3. **Behavior Flow:**
   - Motors initialize and read resting joint angles.
   - The robot smoothly stands up over 2 seconds ($1000$ iterations) to $q_{\text{default}}$.
   - The 50 Hz policy loop activates with $K_p = 40.0, K_d = 4.9$.
   - Steer using keyboard controls (`W/S` for forward/back, `A/D` for lateral, `Left/Right` arrow keys for yaw rotation).
   - Press `Ctrl+C` at any time to softly damp all joints into resting position.

---

## Acknowledgements & References

- [Genesis](https://genesis-world.org/) — Physics platform for generative robotics and reinforcement learning.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — Fast, GPU-accelerated implementation of PPO from the Robotic Systems Lab (ETH Zurich).
- [Unitree Robotics](https://www.unitree.com/) — Unitree Go2 quadruped platform and [`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2_python).
- [Robotics Association of Nepal (RAN)](https://www.ran.org.np/) — Community robotics support and open hardware initiative.

---

## License

This project is licensed under the [MIT License](LICENSE).