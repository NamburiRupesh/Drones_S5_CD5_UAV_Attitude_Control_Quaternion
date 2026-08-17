import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# Desired attitude
# ------------------------------------------------------------

ROLL_DEG = 30.0
PITCH_DEG = 60.0
YAW_DEG = 90.0


# ------------------------------------------------------------
# Quaternion P² controller gains
#
# These are intentionally gentle for this MuJoCo model.
# ------------------------------------------------------------

Pq = 0.01
Pw = 0.005


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

SIM_TIME = 10.0
DT = 0.002


# ------------------------------------------------------------
# Torque saturation
# ------------------------------------------------------------

TORQUE_LIMIT = 0.01


# ============================================================
# XML PATH
# ============================================================

XML_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "quadrotor.xml"
)

if not os.path.exists(XML_PATH):

    raise FileNotFoundError(
        "\nCould not find quadrotor.xml\n\n"
        f"Expected:\n{XML_PATH}\n"
    )


# ============================================================
# QUATERNION FUNCTIONS
# ============================================================

def normalize_quaternion(q):
    """
    Normalize quaternion [w, x, y, z].
    """

    q = np.asarray(q, dtype=float)

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / norm


def quaternion_conjugate(q):
    """
    Quaternion conjugate.

    q = [w, x, y, z]
    """

    return np.array([
        q[0],
        -q[1],
        -q[2],
        -q[3]
    ])


def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication.

    Quaternion format:

        q = [w, x, y, z]
    """

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


# ============================================================
# RPY -> QUATERNION
# ============================================================

def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw [rad]
    to quaternion [w, x, y, z].
    """

    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)

    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)

    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    qw = (
        cr * cp * cy
        + sr * sp * sy
    )

    qx = (
        sr * cp * cy
        - cr * sp * sy
    )

    qy = (
        cr * sp * cy
        + sr * cp * sy
    )

    qz = (
        cr * cp * sy
        - sr * sp * cy
    )

    q = np.array([
        qw,
        qx,
        qy,
        qz
    ])

    return normalize_quaternion(q)


# ============================================================
# QUATERNION -> RPY
# ============================================================

def quaternion_to_rpy(q):
    """
    Convert quaternion [w, x, y, z]
    to roll, pitch, yaw [rad].
    """

    q = normalize_quaternion(q)

    qw, qx, qy, qz = q

    # --------------------------------------------------------
    # Roll
    # --------------------------------------------------------

    sinr_cosp = 2.0 * (
        qw * qx +
        qy * qz
    )

    cosr_cosp = 1.0 - 2.0 * (
        qx * qx +
        qy * qy
    )

    roll = np.arctan2(
        sinr_cosp,
        cosr_cosp
    )

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    sinp = 2.0 * (
        qw * qy -
        qz * qx
    )

    if abs(sinp) >= 1.0:

        pitch = np.sign(sinp) * (
            np.pi / 2.0
        )

    else:

        pitch = np.arcsin(sinp)

    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    siny_cosp = 2.0 * (
        qw * qz +
        qx * qy
    )

    cosy_cosp = 1.0 - 2.0 * (
        qy * qy +
        qz * qz
    )

    yaw = np.arctan2(
        siny_cosp,
        cosy_cosp
    )

    return np.array([
        roll,
        pitch,
        yaw
    ])


# ============================================================
# QUATERNION ATTITUDE ERROR
# ============================================================

def compute_quaternion_error(q_ref, q_current):
    """
    Quaternion attitude error:

        q_err = q_ref ⊗ conjugate(q_current)

    The vector part is used as the attitude error.
    """

    q_ref = normalize_quaternion(q_ref)

    q_current = normalize_quaternion(
        q_current
    )

    q_err = quaternion_multiply(
        q_ref,
        quaternion_conjugate(q_current)
    )

    # --------------------------------------------------------
    # Choose the closest quaternion representation.
    #
    # q and -q represent the same physical orientation.
    # --------------------------------------------------------

    if q_err[0] < 0.0:

        q_err = -q_err

    axis_error = q_err[1:4]

    return q_err, axis_error


# ============================================================
# P² CONTROLLER
# ============================================================

def compute_control_torque(
    q_ref,
    q_current,
    omega
):
    """
    Quaternion P² attitude controller.

    IMPORTANT:
    The sign here has been selected according to the
    experimentally verified MuJoCo torque convention.

        torque = +Pq * quaternion_error
                 -Pw * angular_velocity

    Positive roll error therefore produces positive Mx.
    """

    q_error, axis_error = (
        compute_quaternion_error(
            q_ref,
            q_current
        )
    )

    omega = np.asarray(
        omega,
        dtype=float
    )

    # --------------------------------------------------------
    # Corrected feedback sign
    # --------------------------------------------------------

    torque = (
        Pq * axis_error
        - Pw * omega
    )

    # --------------------------------------------------------
    # Torque saturation
    # --------------------------------------------------------

    torque = np.clip(
        torque,
        -TORQUE_LIMIT,
        TORQUE_LIMIT
    )

    return torque, q_error


# ============================================================
# DESIRED ATTITUDE
# ============================================================

roll_ref = np.deg2rad(
    ROLL_DEG
)

pitch_ref = np.deg2rad(
    PITCH_DEG
)

yaw_ref = np.deg2rad(
    YAW_DEG
)


q_ref = rpy_to_quaternion(
    roll_ref,
    pitch_ref,
    yaw_ref
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print()
print("==============================================")
print(" QUATERNION P² CONTROLLER - MuJoCo VIEWER")
print("==============================================")

print()

print("XML:")
print(XML_PATH)

print()

print("Desired attitude:")
print(
    f"Roll  : {ROLL_DEG:.2f} deg"
)

print(
    f"Pitch : {PITCH_DEG:.2f} deg"
)

print(
    f"Yaw   : {YAW_DEG:.2f} deg"
)

print()

print("Reference quaternion:")
print(
    q_ref
)

print()

print(
    f"Pq = {Pq}"
)

print(
    f"Pw = {Pw}"
)

print(
    f"Torque limit = ±{TORQUE_LIMIT} Nm"
)

print(
    f"Simulation time = {SIM_TIME} s"
)

print(
    f"DT = {DT} s"
)

print()


# ============================================================
# LOAD MUJOCO
# ============================================================

model = mujoco.MjModel.from_xml_path(
    XML_PATH
)

data = mujoco.MjData(
    model
)

model.opt.timestep = DT


# ============================================================
# FIND QUADROTOR BODY
# ============================================================

drone_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "quadrotor"
)

if drone_body_id < 0:

    raise RuntimeError(
        "Could not find body named 'quadrotor' "
        "in quadrotor.xml"
    )


print(
    f"Drone body ID: {drone_body_id}"
)

print()


# ============================================================
# STORAGE
# ============================================================

time_history = []

rpy_history = []

torque_history = []

q_history = []

error_history = []


# ============================================================
# INITIALIZE SIMULATION
# ============================================================

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# OPEN MUJOCO VIEWER
# ============================================================

print("==============================================")
print(" Opening MuJoCo viewer...")
print("==============================================")

print()

print(
    "The drone should smoothly move toward:"
)

print(
    f"Roll  = {ROLL_DEG:.2f}°"
)

print(
    f"Pitch = {PITCH_DEG:.2f}°"
)

print(
    f"Yaw   = {YAW_DEG:.2f}°"
)

print()

print(
    "Keep the viewer open to watch the simulation."
)

print(
    "Close the viewer window when finished."
)

print()


# ============================================================
# REAL-TIME MUJOCO SIMULATION
# ============================================================

wall_start = time.perf_counter()

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    # --------------------------------------------------------
    # Camera setup
    # --------------------------------------------------------

    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0

    # --------------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------------

    steps = int(
        SIM_TIME / DT
    )

    for step in range(steps):

        # ----------------------------------------------------
        # Stop if viewer was closed
        # ----------------------------------------------------

        if not viewer.is_running():

            print()
            print(
                "Viewer closed by user."
            )

            break

        # ----------------------------------------------------
        # Simulation time
        # ----------------------------------------------------

        t = step * DT

        # ----------------------------------------------------
        # Current quaternion
        #
        # MuJoCo free-joint qpos:
        #
        # qpos[0:3] = position
        # qpos[3:7] = quaternion
        # ----------------------------------------------------

        q_current = np.array(
            data.qpos[3:7],
            dtype=float
        )

        q_current = normalize_quaternion(
            q_current
        )

        # ----------------------------------------------------
        # Angular velocity
        #
        # Free-joint qvel:
        #
        # qvel[0:3] = linear velocity
        # qvel[3:6] = angular velocity
        # ----------------------------------------------------

        omega = np.array(
            data.qvel[3:6],
            dtype=float
        )

        # ----------------------------------------------------
        # Quaternion controller
        # ----------------------------------------------------

        torque, q_error = (
            compute_control_torque(
                q_ref,
                q_current,
                omega
            )
        )

        # ----------------------------------------------------
        # Extra safety saturation
        # ----------------------------------------------------

        torque = np.clip(
            torque,
            -TORQUE_LIMIT,
            TORQUE_LIMIT
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Apply torque to the actual quadrotor body.
        #
        # body ID = drone_body_id
        #
        # xfrc_applied:
        #
        # [0:3] = force
        # [3:6] = torque
        # ----------------------------------------------------

        data.xfrc_applied[
            drone_body_id,
            0:3
        ] = 0.0

        data.xfrc_applied[
            drone_body_id,
            3:6
        ] = torque

        # ----------------------------------------------------
        # Convert current attitude to RPY
        # ----------------------------------------------------

        rpy = quaternion_to_rpy(
            q_current
        )

        # ----------------------------------------------------
        # Store data
        # ----------------------------------------------------

        time_history.append(
            t
        )

        rpy_history.append(
            rpy.copy()
        )

        torque_history.append(
            torque.copy()
        )

        q_history.append(
            q_current.copy()
        )

        error_history.append(
            q_error.copy()
        )

        # ----------------------------------------------------
        # Step MuJoCo
        # ----------------------------------------------------

        mujoco.mj_step(
            model,
            data
        )

        # ----------------------------------------------------
        # Update viewer
        # ----------------------------------------------------

        viewer.sync()

        # ----------------------------------------------------
        # Real-time pacing
        # ----------------------------------------------------

        target_wall_time = (
            wall_start + t
        )

        remaining = (
            target_wall_time -
            time.perf_counter()
        )

        if remaining > 0:

            time.sleep(
                remaining
            )


# ============================================================
# CONVERT DATA TO NUMPY
# ============================================================

time_history = np.asarray(
    time_history
)

rpy_history = np.asarray(
    rpy_history
)

torque_history = np.asarray(
    torque_history
)

q_history = np.asarray(
    q_history
)

error_history = np.asarray(
    error_history
)


# ============================================================
# HANDLE EMPTY SIMULATION
# ============================================================

if len(time_history) == 0:

    print()
    print(
        "No simulation data was recorded."
    )

    sys.exit(0)


# ============================================================
# FINAL ATTITUDE
# ============================================================

final_rpy = rpy_history[-1]

final_roll = np.rad2deg(
    final_rpy[0]
)

final_pitch = np.rad2deg(
    final_rpy[1]
)

final_yaw = np.rad2deg(
    final_rpy[2]
)


# ============================================================
# ATTITUDE ERROR
# ============================================================

rpy_reference = np.array([
    ROLL_DEG,
    PITCH_DEG,
    YAW_DEG
])


rpy_output_deg = np.rad2deg(
    rpy_history
)


rpy_error_deg = (
    rpy_reference -
    rpy_output_deg
)


final_error = (
    rpy_error_deg[-1]
)


rms_error = np.sqrt(
    np.mean(
        rpy_error_deg ** 2,
        axis=0
    )
)


# ============================================================
# TORQUE STATISTICS
# ============================================================

max_torque = np.max(
    np.abs(torque_history),
    axis=0
)


# ============================================================
# FINAL QUATERNION
# ============================================================

final_q = q_history[-1]


# ============================================================
# RESULTS
# ============================================================

print()
print("==============================================")
print(" MuJoCo RPY ATTITUDE CONTROL COMPLETE")
print("==============================================")

print()

print("Final attitude:")

print(
    f"Roll  : {final_roll:+.4f} deg"
)

print(
    f"Pitch : {final_pitch:+.4f} deg"
)

print(
    f"Yaw   : {final_yaw:+.4f} deg"
)

print()

print("Reference attitude:")

print(
    f"Roll  : {ROLL_DEG:+.4f} deg"
)

print(
    f"Pitch : {PITCH_DEG:+.4f} deg"
)

print(
    f"Yaw   : {YAW_DEG:+.4f} deg"
)

print()

print("Final attitude error:")

print(
    f"Roll  : {final_error[0]:+.4f} deg"
)

print(
    f"Pitch : {final_error[1]:+.4f} deg"
)

print(
    f"Yaw   : {final_error[2]:+.4f} deg"
)

print()

print("RMS attitude error:")

print(
    f"Roll  : {rms_error[0]:.4f} deg"
)

print(
    f"Pitch : {rms_error[1]:.4f} deg"
)

print(
    f"Yaw   : {rms_error[2]:.4f} deg"
)

print()

print(
    f"Maximum |Mx| = "
    f"{max_torque[0]:.6f} Nm"
)

print(
    f"Maximum |My| = "
    f"{max_torque[1]:.6f} Nm"
)

print(
    f"Maximum |Mz| = "
    f"{max_torque[2]:.6f} Nm"
)

print()

print("Final quaternion:")

print(
    f"[{final_q[0]:+.6f}, "
    f"{final_q[1]:+.6f}, "
    f"{final_q[2]:+.6f}, "
    f"{final_q[3]:+.6f}]"
)

print()

print("Simulation completed.")


# ============================================================
# PLOT 1 — RPY ATTITUDE TRACKING
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.subplot(3, 1, 1)

plt.plot(
    time_history,
    rpy_output_deg[:, 0],
    label="Actual"
)

plt.axhline(
    ROLL_DEG,
    linestyle="--",
    label="Reference"
)

plt.ylabel(
    "Roll [deg]"
)

plt.title(
    "Quaternion P² Controller - RPY Attitude Control"
)

plt.grid()
plt.legend()


plt.subplot(3, 1, 2)

plt.plot(
    time_history,
    rpy_output_deg[:, 1],
    label="Actual"
)

plt.axhline(
    PITCH_DEG,
    linestyle="--",
    label="Reference"
)

plt.ylabel(
    "Pitch [deg]"
)

plt.grid()
plt.legend()


plt.subplot(3, 1, 3)

plt.plot(
    time_history,
    rpy_output_deg[:, 2],
    label="Actual"
)

plt.axhline(
    YAW_DEG,
    linestyle="--",
    label="Reference"
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Yaw [deg]"
)

plt.grid()
plt.legend()

plt.tight_layout()


# ============================================================
# PLOT 2 — CONTROL TORQUE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    time_history,
    torque_history[:, 0],
    label="Mx"
)

plt.plot(
    time_history,
    torque_history[:, 1],
    label="My"
)

plt.plot(
    time_history,
    torque_history[:, 2],
    label="Mz"
)

plt.axhline(
    TORQUE_LIMIT,
    linestyle="--",
    label=f"+{TORQUE_LIMIT} Nm"
)

plt.axhline(
    -TORQUE_LIMIT,
    linestyle="--",
    label=f"-{TORQUE_LIMIT} Nm"
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Torque [Nm]"
)

plt.title(
    "Quaternion P² Controller - RPY Control Torque"
)

plt.grid()
plt.legend()

plt.tight_layout()


# ============================================================
# PLOT 3 — QUATERNION RESPONSE
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.subplot(4, 1, 1)

plt.plot(
    time_history,
    q_history[:, 0],
    label="q0"
)

plt.axhline(
    q_ref[0],
    linestyle="--",
    label="q0 Reference"
)

plt.ylabel(
    "q0"
)

plt.grid()
plt.legend()


plt.subplot(4, 1, 2)

plt.plot(
    time_history,
    q_history[:, 1],
    label="q1"
)

plt.axhline(
    q_ref[1],
    linestyle="--",
    label="q1 Reference"
)

plt.ylabel(
    "q1"
)

plt.grid()
plt.legend()


plt.subplot(4, 1, 3)

plt.plot(
    time_history,
    q_history[:, 2],
    label="q2"
)

plt.axhline(
    q_ref[2],
    linestyle="--",
    label="q2 Reference"
)

plt.ylabel(
    "q2"
)

plt.grid()
plt.legend()


plt.subplot(4, 1, 4)

plt.plot(
    time_history,
    q_history[:, 3],
    label="q3"
)

plt.axhline(
    q_ref[3],
    linestyle="--",
    label="q3 Reference"
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "q3"
)

plt.grid()
plt.legend()

plt.tight_layout()


# ============================================================
# SHOW PLOTS
# ============================================================

plt.show()