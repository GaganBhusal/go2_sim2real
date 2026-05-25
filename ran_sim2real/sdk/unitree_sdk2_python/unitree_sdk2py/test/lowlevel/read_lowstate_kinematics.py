"""
Unitree Go2 – Live Velocity Visualiser  (improved)
====================================================
Windows
  1. Base Linear Velocity  (vx, vy, vz)
  2. Base Angular Velocity (wx, wy, wz)  ← NEW: from gyroscope directly
  3. Foot Velocity Magnitude per leg
  4. IMU Roll / Pitch / Yaw (integrated)

Math improvements
  - Linear velocity: proper numerical integration of IMU accelerometer with
    a first-order high-pass filter to suppress DC drift.
  - Angular velocity: taken directly from the gyroscope (no integration needed).
  - Foot velocity: proper Jacobian per joint configuration using the Go2 link
    lengths and the actual joint angles q[], not a fixed placeholder.
"""

import time
import math
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


# ─────────────────────────────────────────────
#  ROBOT CONSTANTS  (Go2 URDF link lengths, m)
# ─────────────────────────────────────────────
L_HIP   = 0.0955   # hip offset  (lateral)
L_THIGH = 0.213    # thigh length
L_CALF  = 0.213    # calf  length

# ─────────────────────────────────────────────
#  LEG LAYOUT
# ─────────────────────────────────────────────
LEG_GROUPS = {
    "FR": [0, 1, 2],
    "FL": [3, 4, 5],
    "RR": [6, 7, 8],
    "RL": [9, 10, 11],
}

# ─────────────────────────────────────────────
#  BUFFERS
# ─────────────────────────────────────────────
MAX_LEN = 400
t_buf = deque(maxlen=MAX_LEN)

# linear velocity (integrated + HP filtered)
lin_vx = deque(maxlen=MAX_LEN)
lin_vy = deque(maxlen=MAX_LEN)
lin_vz = deque(maxlen=MAX_LEN)

# angular velocity (direct from gyro)
ang_wx = deque(maxlen=MAX_LEN)
ang_wy = deque(maxlen=MAX_LEN)
ang_wz = deque(maxlen=MAX_LEN)

# foot velocity magnitude
foot_bufs = {leg: deque(maxlen=MAX_LEN) for leg in LEG_GROUPS}

# orientation (integrated from gyro)
roll_buf  = deque(maxlen=MAX_LEN)
pitch_buf = deque(maxlen=MAX_LEN)
yaw_buf   = deque(maxlen=MAX_LEN)

# ─────────────────────────────────────────────
#  INTEGRATION STATE
# ─────────────────────────────────────────────
_prev_t   = None
_vel      = np.zeros(3)     # current integrated velocity
_hp_vel   = np.zeros(3)     # high-pass filtered velocity
_prev_acc = np.zeros(3)
_euler    = np.zeros(3)     # roll, pitch, yaw

HP_ALPHA  = 0.98            # high-pass time constant (0 = no integration, 1 = pure integration)
G_VEC     = np.array([0.0, 0.0, 9.81])   # gravity in body frame at rest (approx)

start_time = time.time()


# ─────────────────────────────────────────────
#  FOOT JACOBIAN  (simplified 3-DOF per leg)
# ─────────────────────────────────────────────
def foot_velocity_body_frame(q: list[float], dq: list[float], side: int) -> np.ndarray:
    """
    Compute foot velocity in the body frame for one leg.

    q   : [q_hip_roll, q_hip_pitch, q_knee_pitch]
    dq  : joint velocities matching q
    side: +1 for right legs (FR, RR), -1 for left legs (FL, RL)

    The Jacobian is derived from the forward kinematics:
        p_foot = R_hip_roll  * (p_hip + R_hip_pitch * (p_thigh + R_knee * p_calf))
    We use a column-wise Jacobian J where  v_foot = J @ dq.
    """
    q0, q1, q2 = float(q[0]), float(q[1]), float(q[2])
    s0, c0 = math.sin(q0), math.cos(q0)
    s1, c1 = math.sin(q1), math.cos(q1)
    s12 = math.sin(q1 + q2)
    c12 = math.cos(q1 + q2)

    lt, lc = L_THIGH, L_CALF

    # foot position components (z-down convention, x-forward)
    # px = lt*s1 + lc*s12  (forward)
    # py = side * L_HIP * c0 - (lt*c1 + lc*c12) * s0  (lateral)
    # pz = -L_HIP * s0 * side - (lt*c1 + lc*c12) * c0  (vertical, negative = down)
    # — partial derivatives give the Jacobian columns:

    # d/dq0  (hip roll)
    dpx_dq0 = 0.0
    dpy_dq0 = -side * L_HIP * s0 - (lt * c1 + lc * c12) * c0
    dpz_dq0 = -side * L_HIP * c0 + (lt * c1 + lc * c12) * s0

    # d/dq1  (hip pitch)
    dpx_dq1 = lt * c1 + lc * c12
    dpy_dq1 = (lt * s1 + lc * s12) * s0
    dpz_dq1 = (lt * s1 + lc * s12) * c0

    # d/dq2  (knee pitch)
    dpx_dq2 = lc * c12
    dpy_dq2 = lc * s12 * s0
    dpz_dq2 = lc * s12 * c0

    J = np.array([
        [dpx_dq0, dpx_dq1, dpx_dq2],
        [dpy_dq0, dpy_dq1, dpy_dq2],
        [dpz_dq0, dpz_dq1, dpz_dq2],
    ])

    return J @ np.array(dq, dtype=float)


# ─────────────────────────────────────────────
#  CALLBACK
# ─────────────────────────────────────────────
def LowStateHandler(msg: LowState_):
    global _prev_t, _vel, _hp_vel, _prev_acc, _euler

    t_now = time.time() - start_time
    t_buf.append(t_now)

    # ── Raw IMU ──────────────────────────────
    acc  = np.array(msg.imu_state.accelerometer, dtype=float)   # m/s²
    gyro = np.array(msg.imu_state.gyroscope,     dtype=float)   # rad/s
    quat = msg.imu_state.quaternion                              # [w,x,y,z]

    # ── Angular velocity (direct, no integration needed) ──
    ang_wx.append(gyro[0])
    ang_wy.append(gyro[1])
    ang_wz.append(gyro[2])

    # ── Orientation from quaternion (if available & non-zero) ──
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    if abs(w) + abs(x) + abs(y) + abs(z) > 0.01:
        # proper quaternion → Euler
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll  = math.atan2(sinr, cosr)
        sinp  = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny  = 2.0 * (w * z + x * y)
        cosy  = 1.0 - 2.0 * (y * y + z * z)
        yaw   = math.atan2(siny, cosy)
    else:
        # fallback: integrate gyro
        dt = (1.0 / 50.0) if _prev_t is None else max(1e-4, t_now - _prev_t)
        _euler += gyro * dt
        roll, pitch, yaw = _euler

    roll_buf.append(math.degrees(roll))
    pitch_buf.append(math.degrees(pitch))
    yaw_buf.append(math.degrees(yaw))

    # ── Linear velocity: integrate gravity-compensated acceleration ──
    dt = (1.0 / 50.0) if _prev_t is None else max(1e-4, t_now - _prev_t)
    _prev_t = t_now

    # Gravity compensation in body frame (approx, works well when near level)
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    g_body = np.array([
        -sp,
        sr * cp,
        cr * cp,
    ]) * 9.81

    acc_corrected = acc - g_body

    # Integrate
    _vel += acc_corrected * dt

    # High-pass filter to remove drift (leaky integrator)
    _hp_vel = HP_ALPHA * (_hp_vel + acc_corrected * dt)

    lin_vx.append(_hp_vel[0])
    lin_vy.append(_hp_vel[1])
    lin_vz.append(_hp_vel[2])

    # ── Foot velocities ──────────────────────
    sides = {"FR": +1, "FL": -1, "RR": +1, "RL": -1}
    for leg, indices in LEG_GROUPS.items():
        q_leg  = [msg.motor_state[i].q  for i in indices]
        dq_leg = [msg.motor_state[i].dq for i in indices]
        v_foot = foot_velocity_body_frame(q_leg, dq_leg, sides[leg])
        foot_bufs[leg].append(float(np.linalg.norm(v_foot)))


# ─────────────────────────────────────────────
#  PLOTTING  (3 separate windows)
# ─────────────────────────────────────────────
plt.ion()

# ── Window 1: Linear + Angular velocity ──
fig1, axs1 = plt.subplots(2, 1, figsize=(11, 7))
fig1.suptitle("Base Velocity", fontsize=13, fontweight="bold")
fig1.canvas.manager.set_window_title("Window 1 – Base Velocity")

# ── Window 2: Foot velocities ──
fig2, ax2 = plt.subplots(1, 1, figsize=(11, 4))
fig2.suptitle("Foot Velocity Magnitude", fontsize=13, fontweight="bold")
fig2.canvas.manager.set_window_title("Window 2 – Foot Velocities")

# ── Window 3: Orientation ──
fig3, ax3 = plt.subplots(1, 1, figsize=(11, 4))
fig3.suptitle("IMU Orientation (deg)", fontsize=13, fontweight="bold")
fig3.canvas.manager.set_window_title("Window 3 – Orientation")

COLORS_LIN = ["#e74c3c", "#2ecc71", "#3498db"]   # vx vy vz
COLORS_ANG = ["#f39c12", "#9b59b6", "#1abc9c"]   # wx wy wz
COLORS_FOOT = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


def _snap(buf):
    """Return a plain list snapshot of a deque (atomic copy)."""
    return list(buf)


def update_plots():
    # ── Snapshot ALL buffers at once before doing any length checks ──
    # This prevents the race condition where the callback appends between reads,
    # causing x (t) and y (data) arrays to have different lengths.
    t      = _snap(t_buf)
    s_lvx  = _snap(lin_vx);  s_lvy = _snap(lin_vy);  s_lvz = _snap(lin_vz)
    s_awx  = _snap(ang_wx);  s_awy = _snap(ang_wy);  s_awz = _snap(ang_wz)
    s_roll = _snap(roll_buf); s_pit = _snap(pitch_buf); s_yaw = _snap(yaw_buf)
    s_foot = {leg: _snap(buf) for leg, buf in foot_bufs.items()}

    # Trim every series to the shortest common length so shapes always match
    n = min(len(t),
            len(s_lvx), len(s_lvy), len(s_lvz),
            len(s_awx), len(s_awy), len(s_awz),
            len(s_roll), len(s_pit), len(s_yaw),
            *(len(v) for v in s_foot.values()))

    if n < 2:
        return

    t      = t[-n:]
    s_lvx  = s_lvx[-n:];  s_lvy = s_lvy[-n:];  s_lvz = s_lvz[-n:]
    s_awx  = s_awx[-n:];  s_awy = s_awy[-n:];  s_awz = s_awz[-n:]
    s_roll = s_roll[-n:]; s_pit = s_pit[-n:];   s_yaw = s_yaw[-n:]
    s_foot = {leg: v[-n:] for leg, v in s_foot.items()}

    # ────── Fig 1: linear ──────────────────
    axs1[0].cla()
    for data, label, color in zip(
        [s_lvx, s_lvy, s_lvz], ["vx", "vy", "vz"], COLORS_LIN
    ):
        axs1[0].plot(t, data, label=label, color=color, linewidth=1.5)
    axs1[0].set_ylabel("m/s")
    axs1[0].set_title("Linear Velocity (HP-filtered integration)")
    axs1[0].legend(loc="upper left", ncol=3)
    axs1[0].grid(alpha=0.3)
    axs1[0].axhline(0, color="white", linewidth=0.5, alpha=0.5)

    # ────── Fig 1: angular ─────────────────
    axs1[1].cla()
    for data, label, color in zip(
        [s_awx, s_awy, s_awz], ["ωx (roll)", "ωy (pitch)", "ωz (yaw)"], COLORS_ANG
    ):
        axs1[1].plot(t, data, label=label, color=color, linewidth=1.5)
    axs1[1].set_ylabel("rad/s")
    axs1[1].set_xlabel("time (s)")
    axs1[1].set_title("Angular Velocity (direct gyroscope)")
    axs1[1].legend(loc="upper left", ncol=3)
    axs1[1].grid(alpha=0.3)
    axs1[1].axhline(0, color="white", linewidth=0.5, alpha=0.5)

    fig1.tight_layout()
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    # ────── Fig 2: foot ────────────────────
    ax2.cla()
    for (leg, data), color in zip(s_foot.items(), COLORS_FOOT):
        ax2.plot(t, data, label=leg, color=color, linewidth=1.5)
    ax2.set_ylabel("‖v_foot‖  m/s")
    ax2.set_xlabel("time (s)")
    ax2.set_title("Foot Speed in Body Frame (analytical Jacobian)")
    ax2.legend(loc="upper left", ncol=4)
    ax2.grid(alpha=0.3)

    fig2.tight_layout()
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    # ────── Fig 3: orientation ─────────────
    ax3.cla()
    for data, label, color in zip(
        [s_roll, s_pit, s_yaw], ["roll", "pitch", "yaw"],
        ["#e74c3c", "#2ecc71", "#3498db"]
    ):
        ax3.plot(t, data, label=label, color=color, linewidth=1.5)
    ax3.set_ylabel("degrees")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Orientation (quaternion → Euler, or gyro integration fallback)")
    ax3.legend(loc="upper left", ncol=3)
    ax3.grid(alpha=0.3)
    ax3.axhline(0, color="white", linewidth=0.5, alpha=0.5)

    fig3.tight_layout()
    fig3.canvas.draw()
    fig3.canvas.flush_events()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp4s0")
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(LowStateHandler, 10)

    print("📊  Live plotting started — 3 windows")
    print("     Window 1 : Base linear + angular velocity")
    print("     Window 2 : Foot velocity magnitude (analytical Jacobian)")
    print("     Window 3 : IMU orientation (roll / pitch / yaw)")

    while True:
        update_plots()
        plt.pause(0.04)   # ~25 Hz refresh