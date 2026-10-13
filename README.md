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
      <a href="https://www.youtube.com/shorts/0ZMWlH1WIyI"><b>Watch Demo 1</b></a>
    </td>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/shorts/KFbTj_H0R_Y" target="_blank">
        <img src="https://img.youtube.com/vi/KFbTj_H0R_Y/maxresdefault.jpg" alt="Go2 Hardware Locomotion Demo 2" width="100%" />
      </a>
      <br />
      <a href="https://www.youtube.com/shorts/KFbTj_H0R_Y"><b>Watch Demo 2</b></a>
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
  - [3. Pretrained Policy Playback (go2_walk_seven / model_10000.pt)](#3-pretrained-policy-playback-go2_walk_seven--model_10000pt)
  - [4. Real Hardware Deployment](#4-real-hardware-deployment)
- [Acknowledgements & References](#acknowledgements--references)
- [License](#license)

---

## Overview

End-to-end reinforcement learning framework for dynamic velocity-tracking locomotion on the **Unitree Go2** quadruped robot, transferred **zero-shot** from simulation to physical hardware.

### Key Highlights
- **Simulation:** Vectorized GPU simulation in [Genesis](https://genesis-world.org/) with 4,096 environments at 500 Hz physics (10 substeps per 50 Hz policy step).
- **RL Training:** Asymmetric Actor-Critic (PPO) via [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) with a 47-dim proprioceptive policy and 55-dim privileged critic.
- **Domain Randomization:** Randomization over mass, CoM shift, friction, actuator gains ($K_p, K_d$), and external push impulses for robust sim-to-real transfer.
- **Hardware Deployment:** Direct 500 Hz communication via Unitree Low-Level DDS (`rt/lowcmd` / `rt/lowstate`) with tuned damping ($K_p = 40.0, K_d = 4.9$) and automatic fall-detection safety.

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

#### 1. Policy Actor Observations (47 dimensions)

| Observation | Dim | Scale | Description |
|---|:---:|:---:|---|
| Base Angular Velocity | 3 | 0.50 | 3D body angular velocity (roll, pitch, yaw) from IMU gyroscope |
| Projected Gravity | 3 | 1.00 | Gravity direction vector in robot base frame |
| Joint Position Error | 12 | 1.00 | Joint angles relative to default standing posture |
| Joint Velocities | 12 | 0.05 | Joint angular velocities from motor encoders |
| Previous Actions | 12 | 1.00 | Joint delta targets from previous policy step |
| Commanded Velocity Targets | 3 | 1.00 | User velocity commands in body frame ($v_x, v_y, \omega_z$) |
| Periodic Gait Clock | 2 | 1.00 | Cyclic trotting phase clock ($\sin, \cos$) |

#### 2. Critic Privileged Observations (55 dimensions)

| Observation | Dim | Scale | Description |
|---|:---:|:---:|---|
| Base Linear Velocity | 3 | 2.00 | Ground-truth 3D torso linear velocity in base frame |
| Base Angular Velocity | 3 | 0.50 | Ground-truth 3D torso angular velocity |
| Base Torso Height | 1 | 1.00 | Ground-truth torso height above ground plane |
| Projected Gravity | 3 | 1.00 | Ground-truth gravity vector in base frame |
| Joint Position Errors | 12 | 1.00 | Exact joint angles relative to default posture |
| Joint Angular Velocities | 12 | 0.05 | Exact joint angular velocities |
| Commanded Velocity Targets | 3 | 1.00 | Commanded velocity setpoints ($v_x, v_y, \omega_z$) |
| Feet Flight Air Time | 4 | 1.00 | Swing phase flight duration for each leg |
| Previous Actions | 12 | 1.00 | Preceding joint delta targets from the actor |
| Periodic Gait Clock | 2 | 1.00 | Cyclic trotting phase clock ($\sin, \cos$) |

---

### Action Space

The policy outputs 12-dimensional joint position target deltas added to the nominal standing posture:

```text
target_joint_pos = default_joint_pos + (action * 0.25)
```

- **Action Scale:** `0.25`
- **Nominal Default Posture (hip, thigh, calf):**
  - **Hip:** 0.0° (0.0 rad)
  - **Thigh:** +50.0° (0.8726 rad) for FL/FR, +55.0° (0.9599 rad) for RL/RR
  - **Calf:** -90.0° (-1.5708 rad) for all legs
- **Joint Mechanical Safety Bounds:**
  - **Hip abduction/adduction:** [-20°, +20°]
  - **Thigh flexion/extension:** [-90°, +200°] (front), [-30°, +260°] (rear)
  - **Calf flexion/extension:** [-156°, -48°]

---

### Domain Randomization

To bridge the sim-to-real gap, domain randomization is applied at environment resets and runtime simulation steps:

| Parameter | Randomized Range | Description |
|---|:---:|---|
| **Base Added Mass** | -1.0 to +2.5 kg | Payload and battery weight variations |
| **Center of Mass (CoM) Shift** | ±4 cm (x, y, z) | Asymmetric weight distribution |
| **Ground Friction** | 0.05 to 3.00 | Generalizes across tiles, wood, carpet, and pavement |
| **Joint Damping** | 0.02 to 0.40 N·s/rad | Motor back-EMF and bearing grease viscosity |
| **Joint Friction Loss** | 0.02 to 0.50 N·m | Coulomb friction and gearbox stiction |
| **Rotor Armature Inertia** | 0.001 to 0.008 kg·m² | Reflected motor rotor inertia through gearbox |
| **Joint Stiffness ($K_p$)** | 20.0 to 80.0 N·m/rad | Position tracking stiffness (deployed at 40.0) |
| **Joint Damping ($K_d$)** | 0.20 to 5.00 N·m·s/rad | Damping variation (deployed at 4.9) |
| **Torque Limit Scale** | 70% to 100% | Motor torque scaling and voltage drop margin |
| **External Push Impulses** | 0 to 150 N | External lateral pushes to teach balance recovery |
| **Gyroscope Noise** | std = 0.05 rad/s | MEMS IMU angular rate sensor noise |
| **Gravity Vector Noise** | std = 0.02 | Base orientation estimation uncertainty |
| **Joint Position Noise** | std = 0.01 rad (0.57°) | Encoder resolution limits and link compliance |
| **Joint Velocity Noise** | std = 0.50 rad/s | Numerical differentiation noise from motor state |

---

### Reward Formulation

The reward function balances velocity tracking, gait quality, energy efficiency, and safety constraints across 22 objectives:

| Category | Term | Weight | Description |
|---|---|:---:|---|
| **Command Tracking** | Linear Velocity X | +1.00 | Track commanded forward / backward body velocity |
| | Linear Velocity Y | +1.00 | Track commanded lateral strafing velocity |
| | Yaw Rate Z | +1.00 | Track commanded heading turn rate |
| **Gait & Cadence** | Feet Air Time | +1.50 | Encourage dynamic flight phase (~150 ms target) and prevent dragging |
| | Phase Contact Sync | +0.50 | Synchronize foot contacts with cyclic gait clock |
| | Feet Swing Height | +0.50 | Track desired clearance arc during swing |
| | Foot Slip Penalty | -0.25 | Penalize foot skidding while in contact with ground |
| | Foot Clearance | 0.00 | Secondary obstacle clearance (inactive) |
| **Base Posture** | Base Height | -30.00 | Maintain nominal torso clearance (0.31 m) |
| | Vertical Velocity | -2.00 | Suppress vertical body bobbing and bouncing |
| | Base Angular Rates | -0.05 | Minimize body rocking in roll and pitch |
| | Stand Still Posture | -0.40 | Hold clean upright posture when zero velocity is commanded |
| | Nominal Joint Posture | -0.50 | Regularize joints toward default standing posture |
| | Hip Splay Deviation | -2.00 | Restrict lateral hip splay |
| **Smoothness & Energy** | Action Rate ($\Delta a$) | -0.01 | Penalize high-frequency command jitter |
| | Joint Accelerations | -5e-7 | Suppress abrupt actuator accelerations |
| | Joint Velocities | -1e-3 | Penalize excessive motor spinning speed |
| | Motor Torques | -1e-5 | Minimize power consumption and actuator thermal load |
| **Safety & Limits** | Soft Joint Limits | -0.10 | Soft barrier penalty when joints near range limits |
| | Body Link Collision | -1.00 | Penalize self-collisions and base/thigh ground contacts |
| | Survival Bonus | +1.00 | Constant incentive per alive timestep |
| | Terminal Fall Penalty | -500.00 | Large penalty upon fall, rollover, or base collapse |

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
  --exp_name go2_walk \
  --domain \
  --num_envs 4096 \
  --max_iter 10000

# Resume training from a specific checkpoint
python scripts/train.py \
  --exp_name go2_walk \
  --domain \
  --resume \
  --ckpt 5000
```

Monitor training progress via TensorBoard:
```bash
tensorboard --logdir logs/
```

---

### 3. Pretrained Policy Playback (go2_walk_seven / model_10000.pt)

The trained policy checkpoint from the `go2_walk_seven` training run at iteration 10,000 is stored at `logs/go2_walk_seven/model_10000.pt`. You can evaluate and run this policy across Genesis simulation, MuJoCo (`mjlab`), and physical Go2 hardware:

#### A. Interactive Simulation in Genesis (Keyboard Control)
Run the real-time interactive controller in Genesis:
```bash
python scripts/walk_controller_with_keyboard.py \
  --exp_name go2_walk_seven \
  --ckpt 10000 \
  --device cuda
```

**Keyboard Controls:**
| Key | Action | Envelope |
|---|---|:---:|
| `Up` / `Down` | Forward / Backward Velocity | $v_x \in [-1.0, 1.0]$ m/s |
| `Left` / `Right` | Lateral Left / Right Velocity | $v_y \in [-1.0, 1.0]$ m/s |
| `Numpad 1` / `Numpad 3` (or `1` / `3`) | Heading Turn Rate (Yaw) | $\omega_z \in [-1.0, 1.0]$ rad/s |


#### B. Deploy to Physical Go2 Hardware
Deploy zero-shot onto the real robot via Unitree Low-Level DDS:
```bash
sudo $(which python) unitree_sdk2_python/example/go2/low_level/custom_test_file.py
```
*(The script automatically loads `logs/go2_walk_seven/model_10000.pt`, ramps from resting posture to standing over 2 seconds, and runs the 50 Hz closed-loop policy with $K_p = 40.0, K_d = 4.9$.)*

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
