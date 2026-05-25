import time
import numpy as np
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

import unitree_go2_const as go2


# ============================
# BUFFERS
# ============================
MAX = 300

t_buf = deque(maxlen=MAX)

acc_buf = deque(maxlen=MAX)
gyro_buf = deque(maxlen=MAX)

roll_buf = deque(maxlen=MAX)
pitch_buf = deque(maxlen=MAX)
yaw_buf = deque(maxlen=MAX)

q_buf = {i: deque(maxlen=MAX) for i in range(12)}
dq_buf = {i: deque(maxlen=MAX) for i in range(12)}

vx_buf = deque(maxlen=MAX)
vy_buf = deque(maxlen=MAX)
vz_buf = deque(maxlen=MAX)

base_v = np.zeros(3)
prev_t = None
start_t = time.time()


# ============================
# CALLBACK
# ============================
def LowStateHandler(msg: LowState_):
    global base_v, prev_t

    t = time.time() - start_t
    t_buf.append(t)

    # -------- IMU RAW --------
    acc = np.array(msg.imu_state.accelerometer, dtype=float)
    gyro = np.array(msg.imu_state.gyroscope, dtype=float)
    quat = msg.imu_state.quaternion

    acc_buf.append(acc)
    gyro_buf.append(gyro)

    # -------- RPY from quaternion --------
    w, x, y, z = quat

    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    # ✅ CONVERT TO DEGREES (ONLY CHANGE)
    roll_buf.append(np.degrees(roll))
    pitch_buf.append(np.degrees(pitch))
    yaw_buf.append(np.degrees(yaw))

    # -------- JOINT RAW --------
    for i in range(12):
        q_buf[i].append(msg.motor_state[i].q)
        dq_buf[i].append(msg.motor_state[i].dq)

    # -------- BASE VELOCITY --------
    if prev_t is None:
        prev_t = t
        return

    dt = t - prev_t
    prev_t = t

    acc_world = acc - np.array([0, 0, 9.81])
    base_v += acc_world * dt

    vx_buf.append(base_v[0])
    vy_buf.append(base_v[1])
    vz_buf.append(base_v[2])


# ============================
# PLOTTING SETUP
# ============================
plt.ion()

fig1, ax1 = plt.subplots(2, 1, figsize=(10, 6))
fig1.suptitle("IMU RAW DATA")

fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8))
fig2.suptitle("Orientation (RPY in Degrees)")

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
fig3.suptitle("Base Velocity (Raw Integration)")

fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
fig4.suptitle("Joint Angles (12 DOF)")

fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
fig5.suptitle("Joint Velocities (12 DOF)")


# ============================
# SAFE UPDATE FUNCTION
# ============================
def update():

    if len(t_buf) < 2:
        return

    t = np.array(t_buf)

    acc = np.array(acc_buf)
    gyro = np.array(gyro_buf)

    roll = np.array(roll_buf)
    pitch = np.array(pitch_buf)
    yaw = np.array(yaw_buf)

    vx = np.array(vx_buf)
    vy = np.array(vy_buf)
    vz = np.array(vz_buf)

    q = [np.array(q_buf[i]) for i in range(12)]
    dq = [np.array(dq_buf[i]) for i in range(12)]

    min_len = min(
        len(t),
        len(acc),
        len(gyro),
        len(roll),
        len(pitch),
        len(yaw),
        len(vx),
        len(vy),
        len(vz),
        *[len(q[i]) for i in range(12)],
        *[len(dq[i]) for i in range(12)],
    )

    if min_len < 2:
        return

    t = t[-min_len:]

    acc = acc[-min_len:]
    gyro = gyro[-min_len:]

    roll = roll[-min_len:]
    pitch = pitch[-min_len:]
    yaw = yaw[-min_len:]

    vx = vx[-min_len:]
    vy = vy[-min_len:]
    vz = vz[-min_len:]

    q = [q[i][-min_len:] for i in range(12)]
    dq = [dq[i][-min_len:] for i in range(12)]

    # =========================
    # IMU
    # =========================
    ax1[0].cla()
    ax1[0].plot(t, acc[:, 0], label="ax")
    ax1[0].plot(t, acc[:, 1], label="ay")
    ax1[0].plot(t, acc[:, 2], label="az")
    ax1[0].legend()

    ax1[1].cla()
    ax1[1].plot(t, gyro[:, 0], label="wx")
    ax1[1].plot(t, gyro[:, 1], label="wy")
    ax1[1].plot(t, gyro[:, 2], label="wz")
    ax1[1].legend()

    # =========================
    # RPY (DEGREES)
    # =========================
    ax2[0].cla()
    ax2[0].plot(t, roll, label="roll (deg)")
    ax2[0].legend()

    ax2[1].cla()
    ax2[1].plot(t, pitch, label="pitch (deg)")
    ax2[1].legend()

    ax2[2].cla()
    ax2[2].plot(t, yaw, label="yaw (deg)")
    ax2[2].legend()

    # =========================
    # BASE VELOCITY
    # =========================
    ax3.cla()
    ax3.plot(t, vx, label="vx")
    ax3.plot(t, vy, label="vy")
    ax3.plot(t, vz, label="vz")
    ax3.legend()

    # =========================
    # JOINT ANGLES
    # =========================
    ax4.cla()
    for i in range(12):
        ax4.plot(t, q[i], label=f"q{i}")
    ax4.legend(ncol=4, fontsize=6)

    # =========================
    # JOINT VELOCITIES
    # =========================
    ax5.cla()
    for i in range(12):
        ax5.plot(t, dq[i], label=f"dq{i}")
    ax5.legend(ncol=4, fontsize=6)

    plt.pause(0.001)


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp4s0")

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(LowStateHandler, 10)

    print("LIVE VISUALIZATION STARTED")

    while True:
        update()
        time.sleep(0.01)