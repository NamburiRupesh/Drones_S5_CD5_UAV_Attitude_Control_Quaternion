import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from live_plotter import LiveTelemetryPlotter
except ImportError:
    from simulations.live_plotter import LiveTelemetryPlotter

CONTROL_MODE = "position"
START_PAUSED = False  # Set True to start simulation in paused state
SHOW_LIVE_GRAPH = True  # Enable real-time dynamic plotting window

# --------------------------------------------------------------------------
# DYNAMIC PLOT SELECTION:
# Choose any combination of channels to display live during simulation:
#   "x"      : X Position tracking (Actual vs Desired)
#   "y"      : Y Position tracking (Actual vs Desired)
#   "z"      : Z Altitude tracking (Actual vs Desired + Ground plane)
#   "error"  : 3D Tracking Error ||p - p_des|| (with live Cur/RMS/Max stats)
#   "roll"   : Roll angle in degrees
#   "pitch"  : Pitch angle in degrees
#   "yaw"    : Yaw angle in degrees
#   "thrust" : Thrust command in Newtons
#
# Examples:
#   LIVE_PLOTS = ["z", "error"]            # Altitude & Error (2 plots)
#   LIVE_PLOTS = ["x", "y", "z", "error"]  # Full 3D tracking (4 plots - default)
#   LIVE_PLOTS = ["z", "thrust", "error"]  # Altitude, Thrust & Error (3 plots)
#   LIVE_PLOTS = ["roll", "pitch", "yaw"]  # Attitude only (3 plots)
# --------------------------------------------------------------------------
LIVE_PLOTS = ["roll","pitch"]

#Attitude interpolation mode: "STEP", "LERP", "NLERP", or "SLERP"

ATTITUDE_INTERPOLATION = "LERP"
INTERPOLATION_DURATION = 5.0                                    

SAVE_CSV = True

ROLL_DEG = 0.0
PITCH_DEG = 0.0
YAW_DEG = 0.0

#Quaternion P² controller gains (attitude mode)
Pq = 0.01
Pw = 0.005

TORQUE_LIMIT = 0.01

# --------------------------------------------------------------------------
# WAYPOINTS & GROUND GEOMETRY REFERENCE:
# - Ground plane surface is at Z = 0.0 m.
# - Resting height of drone body on landing feet: Z ≈ 0.157 m.
# - Recommended safe flight range: Z >= 0.3 m.
# - If a waypoint has Z <= 0.0 m:
#   The rigid ground physically stops the drone from penetrating.
#   The controller stays upright and cuts thrust until the trajectory rises again.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 3D BUTTERFLY WAYPOINTS:
# - Parameterized via Temple H. Fay's butterfly curve with 3D dihedral wing elevation.
# - Starts and ends at [0.0, 0.0, 1.0] (matching drone resting position).
# - Sweeps out:
#     1) Lower left wing (Z: 1.2m -> 1.6m)
#     2) Upper left wing (Z: 1.2m -> 2.1m)
#     3) Center body & head (Z: 1.2m)
#     4) Upper right wing (Z: 1.2m -> 2.0m)
#     5) Lower right wing (Z: 1.1m -> 1.6m)
#     6) Returns to home hover at [0.0, 0.0, 1.0].
# --------------------------------------------------------------------------
WAYPOINTS = [
    [ 0.000,  0.000,  1.000],
    [-0.717, -1.038,  1.505],
    [-1.200, -1.020,  1.630],
    [-0.619, -0.293,  1.274],
    [ 0.420,  0.077,  1.171],
    [ 0.520, -0.042,  1.209],
    [-0.621,  0.222,  1.264],
    [-1.754,  1.212,  1.853],
    [-1.708,  2.009,  2.055],
    [-0.804,  1.696,  1.751],
    [-0.141,  0.769,  1.313],
    [ 0.045,  0.548,  1.220],
    [ 0.482,  1.351,  1.574],
    [ 1.413,  2.045,  1.994],
    [ 1.913,  1.626,  2.004],
    [ 1.173,  0.556,  1.519],
    [-0.145, -0.026,  1.059],
    [-0.574,  0.047,  1.230],
    [ 0.256, -0.091,  1.109],
    [ 1.166, -0.805,  1.567],
    [ 1.049, -1.234,  1.648],
    [ 0.264, -0.557,  1.247],
    [-0.112,  0.615,  1.250],
    [ 0.071,  0.871,  1.350],
    [ 0.000,  0.000,  1.000],
]

SEGMENT_DURATIONS = [1.2] * (len(WAYPOINTS) - 1)

DESIRED_YAW_DEG = 0.0

Kp_POS = 12.0
Kd_POS = 7.0
Ki_POS = 2.0  # Integral gain for wind gust & disturbance rejection
MAX_ACCEL = 10.0

Pq_POS = 0.5
Pw_POS = 0.025
TORQUE_LIMIT_POS = 0.5

# --------------------------------------------------------------------------
# AERODYNAMIC WIND DISTURBANCE (World Inertial Frame):
# - WIND_STRENGTH_N: Baseline lateral wind disturbance force in Newtons.
# - WIND_DIRECTION_DEG: Wind compass direction in degrees (horizontal angle):
#     0°   : East (+X)
#     45°  : North-East (from SW to NE, pushing along +X and +Y) [Default]
#     90°  : North (+Y)
#     180° : West (-X)
#     270° : South (-Y)
# --------------------------------------------------------------------------
WIND_STRENGTH_N = 3.5
WIND_DIRECTION_DEG = 270.0

MASS = 1.0                                      
GRAVITY = 9.81            

MAX_THRUST = 20.0                             
MIN_THRUST = 0.0                                                    

SIM_TIME = 10.0                                                     
DT = 0.002

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

def normalize_quaternion(q):


    q = np.asarray(q, dtype=float)

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / norm

def quaternion_conjugate(q):


    return np.array([
        q[0],
        -q[1],
        -q[2],
        -q[3]
    ])

def quaternion_multiply(q1, q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rpy_to_quaternion(roll, pitch, yaw):
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

def quaternion_to_rpy(q):

    q = normalize_quaternion(q)

    qw, qx, qy, qz = q

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

def compute_quaternion_error(q_ref, q_current):

    q_ref = normalize_quaternion(q_ref)

    q_current = normalize_quaternion(
        q_current
    )

    q_err = quaternion_multiply(
        q_ref,
        quaternion_conjugate(q_current)
    )

    if q_err[0] < 0.0:

        q_err = -q_err

    axis_error = q_err[1:4]

    return q_err, axis_error

def compute_control_torque(
    q_ref,
    q_current,
    omega,
    kp=None,
    kd=None,
    torque_lim=None
):

    if kp is None:
        kp = Pq
    if kd is None:
        kd = Pw
    if torque_lim is None:
        torque_lim = TORQUE_LIMIT

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

    torque = (
        kp * axis_error
        - kd * omega
    )

    torque = np.clip(
        torque,
        -torque_lim,
        torque_lim
    )

    return torque, q_error

def lerp_quaternion(q0, q1, t):

    return (1.0 - t) * np.array(q0) + t * np.array(q1)

def nlerp_quaternion(q0, q1, t):

    q = (1.0 - t) * np.array(q0) + t * np.array(q1)
    return normalize_quaternion(q)

def slerp(q0, q1, t):

    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = np.dot(q0, q1)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return normalize_quaternion(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta

    return normalize_quaternion(w0 * q0 + w1 * q1)

def rotation_matrix_to_quaternion(R):

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(
            1.0 + R[0, 0] - R[1, 1] - R[2, 2]
        )
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(
            1.0 + R[1, 1] - R[0, 0] - R[2, 2]
        )
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s

    else:
        s = 2.0 * np.sqrt(
            1.0 + R[2, 2] - R[0, 0] - R[1, 1]
        )
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return normalize_quaternion(
        np.array([w, x, y, z])
    )

class BSplineWaypointTrajectory:
    def __init__(self, waypoints, durations):
        self.waypoints = np.array(waypoints, dtype=float)
        self.durations = list(durations)
        
        # Calculate cumulative time array
        self.t_points = [0.0]
        for d in self.durations:
            self.t_points.append(self.t_points[-1] + d)
        self.t_points = np.array(self.t_points)

        # For N points, degree k is at most N-1.
        # We use cubic B-spline (k=3) for C2-continuous trajectory when N >= 4,
        # or quadratic (k=2) when N = 3.
        k = min(3, len(self.waypoints) - 1)
        
        # Fit a B-Spline that passes exactly through the waypoints
        self.spline_p = make_interp_spline(self.t_points, self.waypoints, k=k)
        
        # Calculate analytic derivatives for velocity and acceleration
        self.spline_v = self.spline_p.derivative(1)
        self.spline_a = self.spline_p.derivative(2) if k >= 2 else None

    @property
    def total_duration(self):
        return self.t_points[-1]

    def evaluate(self, t):
        if t <= 0.0:
            return self.waypoints[0].copy(), np.zeros(3), np.zeros(3)
        if t >= self.total_duration:
            return self.waypoints[-1].copy(), np.zeros(3), np.zeros(3)

        p_d = self.spline_p(t)
        v_d = self.spline_v(t)
        if self.spline_a is not None:
            a_d = self.spline_a(t)
        else:
            a_d = np.zeros(3)

        return p_d, v_d, a_d

def position_controller(p, v, p_d, v_d, a_d, Kp, Kd, max_accel, e_int=None, Ki=0.0):

    e_p = p_d - p
    e_v = v_d - v

    a_cmd = a_d + Kp * e_p + Kd * e_v
    if e_int is not None and Ki > 0.0:
        a_cmd = a_cmd + Ki * e_int

    accel_mag = np.linalg.norm(a_cmd)

    if accel_mag > max_accel:
        a_cmd = a_cmd * (max_accel / accel_mag)

    return a_cmd

def acceleration_to_thrust_and_attitude(
    a_cmd, mass, gravity, desired_yaw
):

    g_vec = np.array([0.0, 0.0, gravity])
    F_des = mass * (a_cmd + g_vec)

    # Ground contact / descent safeguard:
    # A standard quadrotor only produces upward thrust (+z_B).
    # If commanded vertical force is <= 0 (e.g. waypoint is below ground or descending fast),
    # keep the drone upright on its landing gear and cut thrust, rather than flipping upside-down.
    if F_des[2] <= 0.05:
        q_upright = np.array([
            np.cos(desired_yaw / 2.0),
            0.0,
            0.0,
            np.sin(desired_yaw / 2.0)
        ])
        return 0.0, q_upright

    thrust = np.linalg.norm(F_des)

    if thrust < 1e-6:

        return 0.0, np.array([1.0, 0.0, 0.0, 0.0])
    z_B = F_des / thrust

    x_C = np.array([
        np.cos(desired_yaw),
        np.sin(desired_yaw),
        0.0
    ])

    y_B = np.cross(z_B, x_C)
    y_norm = np.linalg.norm(y_B)

    if y_norm < 1e-6:

        x_C_fallback = np.array([
            -np.sin(desired_yaw),
            np.cos(desired_yaw),
            0.0
        ])
        y_B = np.cross(z_B, x_C_fallback)
        y_norm = np.linalg.norm(y_B)

    y_B = y_B / y_norm

    x_B = np.cross(y_B, z_B)

    R = np.column_stack([x_B, y_B, z_B])

    q_des = rotation_matrix_to_quaternion(R)

    return thrust, q_des

# Pre-allocated visual marker constants (zero allocations per frame)
MARKER_GREEN = np.array([0.2, 0.9, 0.3, 0.5], dtype=np.float32)
MARKER_RED = np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32)
TRAIL_CYAN = np.array([0.0, 0.85, 1.0, 0.85], dtype=np.float32)
WIND_ARROW_RGBA = np.array([1.0, 0.85, 0.0, 0.9], dtype=np.float32)
AXIS_RED = np.array([1.0, 0.15, 0.15, 0.90], dtype=np.float32)
AXIS_GREEN = np.array([0.15, 0.95, 0.25, 0.90], dtype=np.float32)
AXIS_BLUE = np.array([0.20, 0.50, 1.00, 0.90], dtype=np.float32)
THRUST_ARROW_RGBA = np.array([1.0, 0.25, 0.05, 0.88], dtype=np.float32)
IDENTITY_3X3 = np.eye(3).flatten().astype(np.float64)

ROTOR_HUBS_BODY = [
    np.array([ 0.45,  0.00, 0.120], dtype=np.float64),
    np.array([-0.45,  0.00, 0.120], dtype=np.float64),
    np.array([ 0.00,  0.45, 0.120], dtype=np.float64),
    np.array([ 0.00, -0.45, 0.120], dtype=np.float64),
]

def add_visual_markers(viewer, waypoints, p_desired, actual_pts,
                       control_mode, show_desired_path=False,
                       drone_pos=None, wind_active=False, wind_force=None,
                       R_body=None, thrust_cmd=0.0, torque=None):

    if control_mode != "position":
        return

    try:
        scn = viewer.user_scn
    except AttributeError:
        return

    scn.ngeom = 0

    # 1. Waypoint target spheres (Green)
    for wp in waypoints:

        if scn.ngeom >= scn.maxgeom:
            break

        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.06, 0.0, 0.0],
            pos=wp if isinstance(wp, np.ndarray) else np.array(wp, dtype=np.float64),
            mat=IDENTITY_3X3,
            rgba=MARKER_GREEN
        )

        scn.ngeom += 1

    # 2. ACTUAL DRONE TRAJECTORY (Cyan line tracing true movement)
    if actual_pts is not None and len(actual_pts) > 1:
        for i in range(len(actual_pts) - 1):

            if scn.ngeom >= scn.maxgeom:
                break

            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.009,
                actual_pts[i],
                actual_pts[i + 1]
            )

            scn.geoms[scn.ngeom].rgba[:] = TRAIL_CYAN

            scn.ngeom += 1

    # 3. Current desired target position (Red sphere)
    if scn.ngeom < scn.maxgeom:

        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.08, 0.0, 0.0],
            pos=p_desired if isinstance(p_desired, np.ndarray) else np.array(p_desired, dtype=np.float64),
            mat=IDENTITY_3X3,
            rgba=MARKER_RED
        )

        scn.ngeom += 1

    # 4. Wind gust vector (Golden-yellow 3D arrow pushing toward the drone)
    if wind_active and wind_force is not None and drone_pos is not None:
        if scn.ngeom < scn.maxgeom:
            w_norm = np.linalg.norm(wind_force)
            if w_norm > 1e-3:
                w_dir = wind_force / w_norm
                p_arrow_start = drone_pos - 0.75 * w_dir
                mujoco.mjv_connector(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.025,
                    p_arrow_start,
                    drone_pos
                )
                scn.geoms[scn.ngeom].rgba[:] = WIND_ARROW_RGBA
                scn.ngeom += 1

    # 5. Drone Body Coordinate Axes (X=Red, Y=Green, Z=Blue)
    if drone_pos is not None and R_body is not None:
        axis_len = 0.35
        axis_width = 0.010
        # X-axis: Forward (Red)
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                axis_width,
                drone_pos,
                drone_pos + axis_len * R_body[:, 0]
            )
            scn.geoms[scn.ngeom].rgba[:] = AXIS_RED
            scn.ngeom += 1

        # Y-axis: Left (Green)
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                axis_width,
                drone_pos,
                drone_pos + axis_len * R_body[:, 1]
            )
            scn.geoms[scn.ngeom].rgba[:] = AXIS_GREEN
            scn.ngeom += 1

        # Z-axis: Up (Blue)
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                axis_width,
                drone_pos,
                drone_pos + axis_len * R_body[:, 2]
            )
            scn.geoms[scn.ngeom].rgba[:] = AXIS_BLUE
            scn.ngeom += 1

    # 6. Four Rotor Thrust Vectors (Live differential thrust output)
    if drone_pos is not None and R_body is not None and thrust_cmd > 0.05:
        d_arm = 0.45
        f_base = thrust_cmd / 4.0
        tau = torque if torque is not None else np.zeros(3)
        df_pitch = tau[1] / (2.0 * d_arm)
        df_roll  = tau[0] / (2.0 * d_arm)
        df_yaw   = tau[2] * 1.5

        rotor_thrusts = [
            max(0.05, f_base - df_pitch + df_yaw),
            max(0.05, f_base + df_pitch + df_yaw),
            max(0.05, f_base + df_roll  - df_yaw),
            max(0.05, f_base - df_roll  - df_yaw),
        ]
        z_body_unit = R_body[:, 2]

        for hub_b, f_i in zip(ROTOR_HUBS_BODY, rotor_thrusts):
            if scn.ngeom >= scn.maxgeom:
                break
            hub_world = drone_pos + R_body @ hub_b
            arrow_len = np.clip(0.055 * f_i, 0.04, 0.45)
            tip_world = hub_world + arrow_len * z_body_unit
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.012,
                hub_world,
                tip_world
            )
            scn.geoms[scn.ngeom].rgba[:] = THRUST_ARROW_RGBA
            scn.ngeom += 1

roll_ref = np.deg2rad(
    ROLL_DEG
)

pitch_ref = np.deg2rad(
    PITCH_DEG
)

yaw_ref = np.deg2rad(
    YAW_DEG
)

q_ref_attitude = rpy_to_quaternion(
    roll_ref,
    pitch_ref,
    yaw_ref
)

desired_yaw_rad = np.deg2rad(DESIRED_YAW_DEG)

if CONTROL_MODE == "position":

    trajectory = BSplineWaypointTrajectory(WAYPOINTS, SEGMENT_DURATIONS)

    SIM_TIME = trajectory.total_duration + 5.0

print()
print("==============================================")
print(" QUATERNION P² CONTROLLER — MuJoCo VIEWER")
print("==============================================")

print()

print(
    f"Control mode: {CONTROL_MODE}"
)

print()

print("XML:")
print(XML_PATH)

print()

if CONTROL_MODE == "attitude":

    print("Desired attitude:")

    print(
        f"  Roll  : {ROLL_DEG:.2f} deg"
    )

    print(
        f"  Pitch : {PITCH_DEG:.2f} deg"
    )

    print(
        f"  Yaw   : {YAW_DEG:.2f} deg"
    )

    print()

    print("Reference quaternion:")
    print(f"  {q_ref_attitude}")

    print()

    print(f"Pq = {Pq}")
    print(f"Pw = {Pw}")
    print(f"Torque limit = ±{TORQUE_LIMIT} Nm")

elif CONTROL_MODE == "position":

    print("Waypoints:")
    for i, wp in enumerate(WAYPOINTS):
        print(f"  [{i}]  {wp}")

    print()

    print("Segment durations:")
    for i, d in enumerate(SEGMENT_DURATIONS):
        print(f"  Segment {i}→{i+1} : {d:.1f} s")

    print()

    print(
        f"Total trajectory duration: "
        f"{trajectory.total_duration:.1f} s"
    )

    print(
        f"Desired yaw: {DESIRED_YAW_DEG:.1f} deg"
    )

    print()

    print("Position controller gains:")
    print(f"  Kp = {Kp_POS}")
    print(f"  Kd = {Kd_POS}")
    print(f"  Max accel = {MAX_ACCEL} m/s²")

    print()

    print("Attitude controller gains (position mode):")
    print(f"  Pq = {Pq_POS}")
    print(f"  Pw = {Pw_POS}")
    print(f"  Torque limit = ±{TORQUE_LIMIT_POS} Nm")

    print()

    print(
        f"Thrust limits: [{MIN_THRUST}, {MAX_THRUST}] N"
    )
    print(
        f"Hover thrust: {MASS * GRAVITY:.2f} N"
    )

print()

print(
    f"Simulation time = {SIM_TIME:.1f} s"
)

print(
    f"DT = {DT} s"
)

print()

model = mujoco.MjModel.from_xml_path(
    XML_PATH
)

data = mujoco.MjData(
    model
)

model.opt.timestep = DT

if CONTROL_MODE == "position":

    model.opt.gravity[:] = [0.0, 0.0, -GRAVITY]

    print(
        f"Gravity enabled: {model.opt.gravity}"
    )

else:

    print(
        f"Gravity: {model.opt.gravity}  (from XML)"
    )

print()

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

time_history = []

rpy_history = []

torque_history = []

q_history = []

error_history = []

pos_history = []
pos_des_history = []

vel_history = []
vel_des_history = []

accel_des_history = []
accel_cmd_history = []

thrust_history = []

q_des_history = []

mujoco.mj_forward(
    model,
    data
)

print("==============================================")
print(" Opening MuJoCo viewer...")
print("==============================================")

print()

if CONTROL_MODE == "attitude":

    print(
        "The drone should smoothly move toward:"
    )

    print(
        f"  Roll  = {ROLL_DEG:.2f}°"
    )

    print(
        f"  Pitch = {PITCH_DEG:.2f}°"
    )

    print(
        f"  Yaw   = {YAW_DEG:.2f}°"
    )

elif CONTROL_MODE == "position":

    print(
        "The drone should follow the waypoint trajectory."
    )

    print(
        "  Green spheres = waypoints"
    )

    print(
        "  Red sphere    = current desired position"
    )

    print(
        "  Cyan line     = actual drone flight path"
    )

print()

print(
    "Keep the viewer open to watch the simulation."
)

print(
    "Close the viewer window when finished."
)

def update_wind_vector(state):
    rad = np.deg2rad(state["wind_direction_deg"])
    mag = state["wind_strength_n"]
    state["wind_force"] = np.array([
        mag * np.cos(rad),
        mag * np.sin(rad),
        0.0
    ], dtype=np.float64)

sim_state = {
    "paused": START_PAUSED,
    "ui_dirty": True,
    "camera_mode": 0,  # 0: Arena Orbit, 1: Third-Person Chase, 2: FPV Cockpit
    "cam_dirty": True,
    "wind_active": False,
    "wind_strength_n": float(WIND_STRENGTH_N),
    "wind_direction_deg": float(WIND_DIRECTION_DEG),
    "wind_force": np.zeros(3, dtype=np.float64),
}
update_wind_vector(sim_state)

def key_callback(keycode):
    # 32 = SPACE, 80/112 = P/p (pause), 83/115 = S/s (start/play), 87/119 = W/w (wind)
    # 67/99 = C/c (camera mode cycle: 1:Orbit -> 2:Chase -> 3:FPV)
    # 61 / 265 = '+' / UP (increase force), 45 / 264 = '-' / DOWN (decrease force)
    # 93 / 262 = ']' / RIGHT (rotate CCW), 91 / 263 = '[' / LEFT (rotate CW)
    if keycode == 32:
        sim_state["paused"] = not sim_state["paused"]
        sim_state["ui_dirty"] = True
    elif keycode in (80, 112):
        if not sim_state["paused"]:
            sim_state["paused"] = True
            sim_state["ui_dirty"] = True
    elif keycode in (83, 115):
        if sim_state["paused"]:
            sim_state["paused"] = False
            sim_state["ui_dirty"] = True
    elif keycode in (86, 118, 67, 99):  # 'V'/'v' (View) or 'C'/'c' (Camera)
        sim_state["camera_mode"] = (sim_state["camera_mode"] + 1) % 3
        sim_state["cam_dirty"] = True
        sim_state["ui_dirty"] = True
        cam_names = ["1. Arena Orbit", "2. Third-Person Chase", "3. FPV Cockpit"]
        print(f"\n[Camera Mode] Switched to: {cam_names[sim_state['camera_mode']]}")
    elif keycode in (87, 119):
        sim_state["wind_active"] = not sim_state["wind_active"]
        sim_state["ui_dirty"] = True
        status = f"ON ({sim_state['wind_strength_n']:.1f}N @ {sim_state['wind_direction_deg']:.0f}°)" if sim_state["wind_active"] else "OFF"
        print(f"\n[Wind Gust] Crosswind Disturbance: {status}")
    elif keycode in (61, 265):  # '+' or '=' or UP arrow
        sim_state["wind_strength_n"] += 0.5
        update_wind_vector(sim_state)
        sim_state["ui_dirty"] = True
        print(f"\n[Wind Gust] Strength increased: {sim_state['wind_strength_n']:.1f} N @ {sim_state['wind_direction_deg']:.0f}°")
    elif keycode in (45, 264):  # '-' or '_' or DOWN arrow
        sim_state["wind_strength_n"] = max(0.0, sim_state["wind_strength_n"] - 0.5)
        update_wind_vector(sim_state)
        sim_state["ui_dirty"] = True
        print(f"\n[Wind Gust] Strength decreased: {sim_state['wind_strength_n']:.1f} N @ {sim_state['wind_direction_deg']:.0f}°")
    elif keycode in (93, 262):  # ']' or RIGHT arrow
        sim_state["wind_direction_deg"] = (sim_state["wind_direction_deg"] + 15.0) % 360.0
        update_wind_vector(sim_state)
        sim_state["ui_dirty"] = True
        print(f"\n[Wind Gust] Direction rotated: {sim_state['wind_strength_n']:.1f} N @ {sim_state['wind_direction_deg']:.0f}°")
    elif keycode in (91, 263):  # '[' or LEFT arrow
        sim_state["wind_direction_deg"] = (sim_state["wind_direction_deg"] - 15.0) % 360.0
        update_wind_vector(sim_state)
        sim_state["ui_dirty"] = True
        print(f"\n[Wind Gust] Direction rotated: {sim_state['wind_strength_n']:.1f} N @ {sim_state['wind_direction_deg']:.0f}°")

print("Viewer Controls:")
print("  [SPACE]      : Start / Pause / Play")
print("  [P]          : Pause")
print("  [S]          : Play / Resume")
print("  [V] / [C]    : Cycle Camera View (1: Orbit -> 2: Chase -> 3: FPV)")
print("  [W]          : Toggle Wind Disturbance ON / OFF")
print("  [+] / [UP]   : Increase Wind Force (+0.5 N)")
print("  [-] / [DOWN] : Decrease Wind Force (-0.5 N)")
print("  []] / [RIGHT]: Rotate Wind Direction (+15° CCW)")
print("  [[] / [LEFT] : Rotate Wind Direction (-15° CW)")
print("  Close the viewer window to finish and view plots.")
print()

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
    show_left_ui=False,
    show_right_ui=False
) as viewer:

    # Ensure default MuJoCo contact point/force visualization stays disabled
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0

    viewer.sync()
    time.sleep(1.0)

    fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_cockpit")

    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0

    if CONTROL_MODE == "position":

        viewer.cam.distance = 6.0
        viewer.cam.elevation = -30.0
        viewer.cam.lookat[:] = [0.5, 0.5, 1.2]

    wall_start = time.perf_counter()

    steps = int(
        SIM_TIME / DT
    )

    actual_trajectory_pts = [np.array(data.qpos[0:3], dtype=float)]

    plotter = LiveTelemetryPlotter(enabled=SHOW_LIVE_GRAPH, channels=LIVE_PLOTS)
    plotter.start()

    pos_integral = np.zeros(3, dtype=float)

    CAM_LABELS = ["Orbit", "Chase", "FPV"]

    step = 0
    while viewer.is_running() and step < steps:

        cam_label = CAM_LABELS[sim_state["camera_mode"]]

        if sim_state["paused"]:
            if sim_state["ui_dirty"]:
                wind_str = f" | [WIND: {sim_state['wind_strength_n']:.1f}N @ {sim_state['wind_direction_deg']:.0f}°]" if sim_state["wind_active"] else ""
                viewer.set_texts([(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    0,
                    "STATUS",
                    f"[PAUSED] | CAM: {cam_label}{wind_str} (SPACE: Play, V/C: Cam, W: Wind, +/-: Force, [/]: Dir)"
                )])
                sim_state["ui_dirty"] = False
                print(f"\n[Viewer] Simulation PAUSED (Press SPACE to Resume)")

            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
            viewer.sync()
            pause_start = time.perf_counter()
            while viewer.is_running() and sim_state["paused"]:
                time.sleep(0.016)
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
                if sim_state["cam_dirty"]:
                    cam_m = sim_state["camera_mode"]
                    if cam_m == 0:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        viewer.cam.distance = 6.0
                        viewer.cam.elevation = -30.0
                        viewer.cam.azimuth = 135.0
                        viewer.cam.lookat[:] = [0.5, 0.5, 1.2]
                    elif cam_m == 1:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        viewer.cam.lookat[:] = data.qpos[0:3]
                        viewer.cam.distance = 2.2
                        viewer.cam.elevation = -14.0
                    elif cam_m == 2 and fpv_cam_id >= 0:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = fpv_cam_id
                    sim_state["cam_dirty"] = False
                    viewer.sync()

            wall_start += (time.perf_counter() - pause_start)
            sim_state["ui_dirty"] = True
            if not viewer.is_running():
                break

        if sim_state["ui_dirty"]:
            wind_str = f" | [WIND: {sim_state['wind_strength_n']:.1f}N @ {sim_state['wind_direction_deg']:.0f}°]" if sim_state["wind_active"] else ""
            viewer.set_texts([(
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                0,
                "STATUS",
                f"[RUNNING] | CAM: {cam_label}{wind_str} (SPACE: Pause, V/C: Cam, W: Wind, +/-: Force, [/]: Dir)"
            )])
            sim_state["ui_dirty"] = False
            print(f"\n[Viewer] Simulation RUNNING | CAM: {cam_label}{wind_str}")

        t = step * DT

        pos = np.array(
            data.qpos[0:3],
            dtype=float
        )

        q_current = normalize_quaternion(
            np.array(
                data.qpos[3:7],
                dtype=float
            )
        )

        vel = np.array(
            data.qvel[0:3],
            dtype=float
        )

        omega = np.array(
            data.qvel[3:6],
            dtype=float
        )

        R_body = data.xmat[
            drone_body_id
        ].reshape(3, 3)

        body_z_world = R_body[:, 2]

        if CONTROL_MODE == "attitude":

            if step == 0:
                q_init = q_current.copy()

            t_interp = np.clip(t / INTERPOLATION_DURATION, 0.0, 1.0)

            if ATTITUDE_INTERPOLATION == "SLERP":
                q_ref_current = slerp(q_init, q_ref_attitude, t_interp)
            elif ATTITUDE_INTERPOLATION == "NLERP":
                q_ref_current = nlerp_quaternion(q_init, q_ref_attitude, t_interp)
            elif ATTITUDE_INTERPOLATION == "LERP":
                q_ref_current = lerp_quaternion(q_init, q_ref_attitude, t_interp)
            else:       
                q_ref_current = q_ref_attitude.copy()

            torque, q_error = (
                compute_control_torque(
                    q_ref_current,
                    q_current,
                    omega
                )
            )

            torque = np.clip(
                torque,
                -TORQUE_LIMIT,
                TORQUE_LIMIT
            )

            data.xfrc_applied[
                drone_body_id,
                0:3
            ] = 0.0

            data.xfrc_applied[
                drone_body_id,
                3:6
            ] = torque

            p_d = pos.copy()
            v_d = vel.copy()
            a_d = np.zeros(3)
            a_cmd = np.zeros(3)
            thrust_cmd = 0.0
            q_des = q_ref_attitude.copy()

        elif CONTROL_MODE == "position":

            p_d, v_d, a_d = trajectory.evaluate(t)

            # Position error integral accumulation with anti-windup clamping
            pos_integral += (p_d - pos) * DT
            pos_integral = np.clip(pos_integral, -1.5, 1.5)

            a_cmd = position_controller(
                pos, vel,
                p_d, v_d, a_d,
                Kp_POS, Kd_POS,
                MAX_ACCEL,
                e_int=pos_integral,
                Ki=Ki_POS
            )

            thrust_cmd, q_des = (
                acceleration_to_thrust_and_attitude(
                    a_cmd,
                    MASS,
                    GRAVITY,
                    desired_yaw_rad
                )
            )

            thrust_cmd = np.clip(
                thrust_cmd,
                MIN_THRUST,
                MAX_THRUST
            )

            torque, q_error = compute_control_torque(
                q_des,
                q_current,
                omega,
                kp=Pq_POS,
                kd=Pw_POS,
                torque_lim=TORQUE_LIMIT_POS
            )

            force_world = thrust_cmd * body_z_world

            # External aerodynamic wind disturbance injection when toggled ON
            if sim_state["wind_active"]:
                applied_force = force_world + sim_state["wind_force"]
            else:
                applied_force = force_world

            data.xfrc_applied[
                drone_body_id,
                0:3
            ] = applied_force

            data.xfrc_applied[
                drone_body_id,
                3:6
            ] = torque

        rpy = quaternion_to_rpy(
            q_current
        )

        time_history.append(t)

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

        if CONTROL_MODE == "attitude":
            q_des_history.append(q_ref_current.copy())
        else:
            q_des_history.append(q_des.copy())

        pos_history.append(pos.copy())
        pos_des_history.append(p_d.copy())
        vel_history.append(vel.copy())
        vel_des_history.append(v_d.copy())
        accel_des_history.append(a_d.copy())
        accel_cmd_history.append(a_cmd.copy())
        thrust_history.append(thrust_cmd)

        mujoco.mj_step(
            model,
            data
        )

        # Record actual trajectory points (record if moved >= 3.5 cm)
        if len(actual_trajectory_pts) == 0 or np.linalg.norm(pos - actual_trajectory_pts[-1]) >= 0.035:
            actual_trajectory_pts.append(np.asarray(pos, dtype=np.float64))

        # Refresh visual markers and sync viewer at 60 FPS (every 8 steps at 500 Hz)
        if step % 8 == 0:
            # Dynamic Camera Mode tracking
            cam_mode = sim_state["camera_mode"]
            if cam_mode == 0:  # Mode 1: Arena Orbit Cam (wide overview)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                if sim_state["cam_dirty"]:
                    viewer.cam.distance = 6.0
                    viewer.cam.elevation = -30.0
                    viewer.cam.azimuth = 135.0
                    viewer.cam.lookat[:] = [0.5, 0.5, 1.2]
                    sim_state["cam_dirty"] = False
            elif cam_mode == 1:  # Mode 2: Third-Person Chase Cam (tracks behind drone)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = pos + 0.15 * R_body[:, 0]
                viewer.cam.distance = 2.2
                viewer.cam.elevation = -14.0
                yaw_deg = np.rad2deg(rpy[2])
                viewer.cam.azimuth = yaw_deg
            elif cam_mode == 2:  # Mode 3: FPV Cockpit Cam (nose view)
                if fpv_cam_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = fpv_cam_id
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat[:] = pos + 2.0 * R_body[:, 0]
                    viewer.cam.distance = 0.05

            add_visual_markers(
                viewer,
                WAYPOINTS,
                p_d,
                actual_trajectory_pts,
                CONTROL_MODE,
                show_desired_path=False,
                drone_pos=pos,
                wind_active=sim_state["wind_active"],
                wind_force=sim_state["wind_force"],
                R_body=R_body,
                thrust_cmd=thrust_cmd,
                torque=torque
            )
            # Neutralize MuJoCo contact point visualization in case 'C' was pressed
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
            viewer.sync()

        # Stream telemetry to live dashboard at ~31 Hz (every 16 steps at 500 Hz)
        if SHOW_LIVE_GRAPH and step % 16 == 0:
            plotter.send(t, pos, p_d, rpy=rpy, thrust=thrust_cmd, wind_active=sim_state["wind_active"], wind_mag=sim_state["wind_strength_n"])

        # Real-time synchronization
        target_wall_time = wall_start + t
        remaining = target_wall_time - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

        step += 1

    if SHOW_LIVE_GRAPH:
        plotter.stop()

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

pos_history = np.asarray(pos_history)
pos_des_history = np.asarray(pos_des_history)
vel_history = np.asarray(vel_history)
vel_des_history = np.asarray(vel_des_history)
accel_des_history = np.asarray(accel_des_history)
accel_cmd_history = np.asarray(accel_cmd_history)
thrust_history = np.asarray(thrust_history)
q_des_history = np.asarray(q_des_history)

if len(time_history) == 0:

    print()
    print(
        "No simulation data was recorded."
    )

    sys.exit(0)

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

final_q = q_history[-1]

print()
print("==============================================")

if CONTROL_MODE == "attitude":
    print(" MuJoCo RPY ATTITUDE CONTROL COMPLETE")
elif CONTROL_MODE == "position":
    print(" MuJoCo POSITION + ATTITUDE CONTROL COMPLETE")

print("==============================================")

print()

print("Final attitude:")

print(
    f"  Roll  : {final_roll:+.4f} deg"
)

print(
    f"  Pitch : {final_pitch:+.4f} deg"
)

print(
    f"  Yaw   : {final_yaw:+.4f} deg"
)

print()

if CONTROL_MODE == "attitude":

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

    print("Reference attitude:")

    print(
        f"  Roll  : {ROLL_DEG:+.4f} deg"
    )

    print(
        f"  Pitch : {PITCH_DEG:+.4f} deg"
    )

    print(
        f"  Yaw   : {YAW_DEG:+.4f} deg"
    )

    print()

    print("Final attitude error:")

    print(
        f"  Roll  : {final_error[0]:+.4f} deg"
    )

    print(
        f"  Pitch : {final_error[1]:+.4f} deg"
    )

    print(
        f"  Yaw   : {final_error[2]:+.4f} deg"
    )

    print()

    print("RMS attitude error:")

    print(
        f"  Roll  : {rms_error[0]:.4f} deg"
    )

    print(
        f"  Pitch : {rms_error[1]:.4f} deg"
    )

    print(
        f"  Yaw   : {rms_error[2]:.4f} deg"
    )

    print()

if CONTROL_MODE == "position":

    final_pos = pos_history[-1]
    final_des = pos_des_history[-1]
    final_pos_error = final_des - final_pos

    pos_error_history = (
        pos_des_history - pos_history
    )

    rms_pos_error = np.sqrt(
        np.mean(
            pos_error_history ** 2,
            axis=0
        )
    )

    print("Final position:")
    print(f"  X : {final_pos[0]:+.4f} m")
    print(f"  Y : {final_pos[1]:+.4f} m")
    print(f"  Z : {final_pos[2]:+.4f} m")

    print()

    print("Final desired position:")
    print(f"  X : {final_des[0]:+.4f} m")
    print(f"  Y : {final_des[1]:+.4f} m")
    print(f"  Z : {final_des[2]:+.4f} m")

    print()

    print("Final position error:")
    print(f"  X : {final_pos_error[0]:+.4f} m")
    print(f"  Y : {final_pos_error[1]:+.4f} m")
    print(f"  Z : {final_pos_error[2]:+.4f} m")
    print(
        f"  |e| : "
        f"{np.linalg.norm(final_pos_error):.4f} m"
    )

    print()

    print("RMS position error:")
    print(f"  X : {rms_pos_error[0]:.4f} m")
    print(f"  Y : {rms_pos_error[1]:.4f} m")
    print(f"  Z : {rms_pos_error[2]:.4f} m")

    print()

    print(
        f"Thrust range: "
        f"[{np.min(thrust_history):.2f}, "
        f"{np.max(thrust_history):.2f}] N"
    )

    print()

max_torque = np.max(
    np.abs(torque_history),
    axis=0
)

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
    f"  [{final_q[0]:+.6f}, "
    f"{final_q[1]:+.6f}, "
    f"{final_q[2]:+.6f}, "
    f"{final_q[3]:+.6f}]"
)

print()

print("Simulation completed.")

if SAVE_CSV:
    import csv
    csv_filename = os.path.join(PROJECT_ROOT, "simulation_data.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([
            "Time_s", "PosX", "PosY", "PosZ", 
            "Roll_deg", "Pitch_deg", "Yaw_deg", 
            "qW", "qX", "qY", "qZ",
            "Mx_Nm", "My_Nm", "Mz_Nm"
        ])

        for i in range(len(time_history)):
            row = [
                time_history[i],
                pos_history[i, 0], pos_history[i, 1], pos_history[i, 2],
                np.rad2deg(rpy_history[i, 0]), np.rad2deg(rpy_history[i, 1]), np.rad2deg(rpy_history[i, 2]),
                q_history[i, 0], q_history[i, 1], q_history[i, 2], q_history[i, 3],
                torque_history[i, 0], torque_history[i, 1], torque_history[i, 2]
            ]
            writer.writerow(row)

    print(f"\nData saved to {csv_filename} for external analysis (Data Driven Simulation).")

if CONTROL_MODE == "attitude":

    rpy_output_deg = np.rad2deg(rpy_history)

    rpy_des_history = np.array([
        quaternion_to_rpy(q) for q in q_des_history
    ])
    rpy_des_deg = np.rad2deg(rpy_des_history)

    plt.figure(
        figsize=(10, 8)
    )

    plt.subplot(3, 1, 1)

    plt.plot(
        time_history,
        rpy_output_deg[:, 0],
        label="Actual"
    )

    plt.plot(
        time_history,
        rpy_des_deg[:, 0],
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

    plt.plot(
        time_history,
        rpy_des_deg[:, 1],
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

    plt.plot(
        time_history,
        rpy_des_deg[:, 2],
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

    plt.figure(
        figsize=(10, 8)
    )

    plt.subplot(4, 1, 1)

    plt.plot(
        time_history,
        q_history[:, 0],
        label="q0"
    )

    plt.plot(
        time_history,
        q_des_history[:, 0],
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

    plt.plot(
        time_history,
        q_des_history[:, 1],
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

    plt.plot(
        time_history,
        q_des_history[:, 2],
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

    plt.plot(
        time_history,
        q_des_history[:, 3],
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

if CONTROL_MODE == "position":

    axis_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 9)
    )

    for i, (ax, label) in enumerate(
        zip(axes, axis_labels)
    ):
        ax.plot(
            time_history,
            pos_history[:, i],
            label=f"{label} actual",
            linewidth=1.5
        )

        ax.plot(
            time_history,
            pos_des_history[:, i],
            "--",
            label=f"{label} desired",
            linewidth=1.5
        )

        ax.set_ylabel(f"{label} [m]")
        ax.grid(True)
        ax.legend()

    axes[0].set_title(
        "Position Tracking — Cascaded Controller"
    )

    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()

    pos_error = pos_des_history - pos_history

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8)
    )

    for i, (ax, label) in enumerate(
        zip(axes, axis_labels)
    ):
        ax.plot(
            time_history,
            pos_error[:, i],
            label=f"e_{label}",
            linewidth=1.2
        )

        ax.set_ylabel(f"Error {label} [m]")
        ax.grid(True)
        ax.legend()

    axes[0].set_title("Position Error")
    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8)
    )

    for i, (ax, label) in enumerate(
        zip(axes, axis_labels)
    ):
        ax.plot(
            time_history,
            vel_history[:, i],
            label=f"V{label} actual",
            linewidth=1.2
        )

        ax.plot(
            time_history,
            vel_des_history[:, i],
            "--",
            label=f"V{label} desired",
            linewidth=1.2
        )

        ax.set_ylabel(f"V{label} [m/s]")
        ax.grid(True)
        ax.legend()

    axes[0].set_title("Velocity Tracking")
    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()

    plt.figure(figsize=(12, 5))

    plt.plot(
        time_history,
        thrust_history,
        label="Thrust",
        linewidth=1.2
    )

    plt.axhline(
        MASS * GRAVITY,
        linestyle="--",
        color="gray",
        label=f"Hover ({MASS * GRAVITY:.1f} N)"
    )

    plt.axhline(
        MAX_THRUST,
        linestyle=":",
        color="red",
        label=f"Max ({MAX_THRUST} N)"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Thrust [N]")
    plt.title("Thrust Command")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    rpy_output_deg = np.rad2deg(rpy_history)

    rpy_des_history = np.array([
        quaternion_to_rpy(q) for q in q_des_history
    ])

    rpy_des_deg = np.rad2deg(rpy_des_history)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8)
    )

    rpy_labels = ["Roll", "Pitch", "Yaw"]

    for i, (ax, label) in enumerate(
        zip(axes, rpy_labels)
    ):
        ax.plot(
            time_history,
            rpy_output_deg[:, i],
            label=f"{label} actual",
            linewidth=1.2
        )

        ax.plot(
            time_history,
            rpy_des_deg[:, i],
            "--",
            label=f"{label} desired",
            linewidth=1.2
        )

        ax.set_ylabel(f"{label} [deg]")
        ax.grid(True)
        ax.legend()

    axes[0].set_title(
        "Attitude During Position Control"
    )

    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()

    plt.figure(figsize=(12, 6))

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
        TORQUE_LIMIT_POS,
        linestyle="--",
        color="red",
        label=f"+{TORQUE_LIMIT_POS} Nm"
    )

    plt.axhline(
        -TORQUE_LIMIT_POS,
        linestyle="--",
        color="red",
        label=f"-{TORQUE_LIMIT_POS} Nm"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title("Control Torque — Position Mode")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    fig = plt.figure(figsize=(10, 8))

    ax3d = fig.add_subplot(
        111,
        projection="3d"
    )

    ax3d.plot(
        pos_history[:, 0],
        pos_history[:, 1],
        pos_history[:, 2],
        label="Actual",
        linewidth=1.5
    )

    ax3d.plot(
        pos_des_history[:, 0],
        pos_des_history[:, 1],
        pos_des_history[:, 2],
        "--",
        label="Desired",
        linewidth=1.5
    )

    wp_arr = np.array(WAYPOINTS)

    ax3d.scatter(
        wp_arr[:, 0],
        wp_arr[:, 1],
        wp_arr[:, 2],
        c="red",
        s=60,
        marker="o",
        label="Waypoints"
    )

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D Trajectory")
    ax3d.legend()

plt.show()
