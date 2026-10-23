import os
import sys
import time
import csv

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

# ==============================================================================
# SECTION 1: HIGH-LEVEL SIMULATION & MISSION CONFIGURATION
# ==============================================================================
CONTROL_MODE = "position"
START_PAUSED = False
INFINITE_LOOP = False
DT = 0.002
SIM_TIME = 10.0

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

# ==============================================================================
# SECTION 2: TRAJECTORY GENERATION CONFIGURATION
# ==============================================================================
TRAJECTORY_MODE = "CUSTOM_EQUATION"   # "CUSTOM_EQUATION", "PRESET_EQUATION", or "WAYPOINTS"
PRESET_SHAPE = "SPIRAL"               # "SPIRAL", "HELIX", "FIGURE8", "BUTTERFLY"
USE_BSPLINE = True                    # For "WAYPOINTS" mode: True = B-Spline, False = Linear
EQUATION_DURATION = 25.0
DESIRED_YAW_DEG = 0.0

WAYPOINTS = [
    [ 0.000,  0.000,  1.000],
    [ 0.053,  0.041,  1.083],
    [ 0.035,  0.129,  1.167],
    [-0.077,  0.185,  1.250],
    [-0.231,  0.133,  1.333],
    [-0.330, -0.044,  1.417],
    [-0.283, -0.283,  1.500],
    [-0.061, -0.463,  1.583],
    [ 0.267, -0.462,  1.667],
    [ 0.554, -0.230,  1.750],
    [ 0.644,  0.173,  1.833],
    [ 0.446,  0.582,  1.917],
    [ 0.000,  0.800,  2.000],
    [-0.528,  0.688,  2.083],
    [-0.902,  0.242,  2.167],
    [-0.924, -0.383,  2.250],
    [-0.533, -0.924,  2.333],
    [ 0.148, -1.124,  2.417],
    [ 0.849, -0.849,  2.500],
    [ 1.256, -0.165,  2.583],
    [ 1.155,  0.667,  2.667],
    [ 0.536,  1.293,  2.750],
    [-0.380,  1.417,  2.833],
    [-1.216,  0.933,  2.917],
    [-1.600,  0.000,  3.000],
]

SEGMENT_DURATIONS = [1.0] * (len(WAYPOINTS) - 1)

# ==============================================================================
# SECTION 3: QUADROTOR PHYSICAL & ACTUATION PARAMETERS
# ==============================================================================
MASS = 1.0
GRAVITY = 9.81
MIN_THRUST = 0.0
MAX_THRUST = 20.0

ROTOR_HUBS_BODY = [
    np.array([ 0.45,  0.00, 0.120], dtype=np.float64),
    np.array([-0.45,  0.00, 0.120], dtype=np.float64),
    np.array([ 0.00,  0.45, 0.120], dtype=np.float64),
    np.array([ 0.00, -0.45, 0.120], dtype=np.float64),
]

# ==============================================================================
# SECTION 4: CONTROLLER GAINS & LIMITS
# ==============================================================================
# 4A. Position Controller (Outer Loop)
Kp_POS = 12.0
Kd_POS = 7.0
Ki_POS = 2.0
MAX_ACCEL = 10.0

# 4B. Attitude Controller (Position Mode Inner Loop)
Pq_POS = 0.5
Pw_POS = 0.025
TORQUE_LIMIT_POS = 0.5

# 4C. Pure Attitude Mode Settings
ROLL_DEG = 0.0
PITCH_DEG = 0.0
YAW_DEG = 0.0
ATTITUDE_INTERPOLATION = "LERP"
INTERPOLATION_DURATION = 5.0
Pq = 0.01
Pw = 0.005
TORQUE_LIMIT = 0.01

# ==============================================================================
# SECTION 5: ENVIRONMENTAL DISTURBANCE (CROSSWIND)
# ==============================================================================
WIND_STRENGTH_N = 3.5
WIND_DIRECTION_DEG = 0.0

# ==============================================================================
# SECTION 6: TELEMETRY & DATA LOGGING CONFIGURATION
# ==============================================================================
LOG_MODE = "AB_ONLY"   # "AB_ONLY", "FULL", or "OFF"
LOG_RATE_HZ = 50       # Telemetry sampling rate (Hz)
SAVE_CSV = True

# ==============================================================================
# SECTION 7: VISUALIZATION & VIEWER STYLING
# ==============================================================================
SHOW_LIVE_GRAPH = True
LIVE_PLOTS = ["roll", "pitch"]

MARKER_GREEN = np.array([0.2, 0.9, 0.3, 0.5], dtype=np.float32)
MARKER_RED = np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32)
TRAIL_CYAN = np.array([0.0, 0.85, 1.0, 0.85], dtype=np.float32)
WIND_ARROW_RGBA = np.array([1.0, 0.85, 0.0, 0.9], dtype=np.float32)
AXIS_RED = np.array([1.0, 0.15, 0.15, 0.90], dtype=np.float32)
AXIS_GREEN = np.array([0.15, 0.95, 0.25, 0.90], dtype=np.float32)
AXIS_BLUE = np.array([0.20, 0.50, 1.00, 0.90], dtype=np.float32)
THRUST_ARROW_RGBA = np.array([1.0, 0.25, 0.05, 0.88], dtype=np.float32)
IDENTITY_3X3 = np.eye(3).flatten().astype(np.float64)

# ==============================================================================
# SECTION 8: MATHEMATICAL TRAJECTORY EQUATION DEFINITIONS
# ==============================================================================
def custom_trajectory(t):
    theta = 0.6 * t
    r = 0.08 * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 1.0 + 0.1 * theta
    return np.array([x, y, z], dtype=float)

def preset_trajectory(t, shape="SPIRAL"):
    if shape == "SPIRAL":
        theta = 0.6 * t
        r = 0.08 * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 1.0 + 0.1 * theta
    elif shape == "HELIX":
        theta = 0.6 * t
        x = 1.2 * np.cos(theta)
        y = 1.2 * np.sin(theta)
        z = 1.0 + 0.08 * t
    elif shape == "FIGURE8":
        omega = 0.4
        x = 1.5 * np.sin(omega * t)
        y = 1.5 * np.sin(omega * t) * np.cos(omega * t)
        z = 1.2 + 0.3 * np.sin(2.0 * omega * t)
    elif shape == "BUTTERFLY":
        theta = 0.35 * t + 3.48437
        r_fay = np.exp(np.cos(theta)) - 2.0 * np.cos(4.0 * theta) + (np.sin(theta / 12.0))**5
        scale = 0.65
        x = np.sin(theta) * r_fay * scale
        y = np.cos(theta) * r_fay * scale
        z = 1.0 + 0.40 * np.sqrt(x**2 + y**2)
    else:
        x, y, z = 0.0, 0.0, 1.0
    return np.array([x, y, z], dtype=float)

# ==============================================================================
# SECTION 9: QUATERNION & 3D ROTATION MATHEMATICS
# ==============================================================================
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

# ==============================================================================
# SECTION 10: TRAJECTORY GENERATION ENGINES (B-SPLINE, LINEAR, ANALYTIC)
# ==============================================================================
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

class LinearWaypointTrajectory:
    def __init__(self, waypoints, durations):
        self.waypoints = np.array(waypoints, dtype=float)
        self.durations = list(durations)
        self.t_points = [0.0]
        for d in self.durations:
            self.t_points.append(self.t_points[-1] + d)
        self.t_points = np.array(self.t_points)

    @property
    def total_duration(self):
        return self.t_points[-1]

    def evaluate(self, t):
        if t <= 0.0:
            return self.waypoints[0].copy(), np.zeros(3), np.zeros(3)
        if t >= self.total_duration:
            return self.waypoints[-1].copy(), np.zeros(3), np.zeros(3)

        idx = np.searchsorted(self.t_points, t, side="right") - 1
        idx = min(idx, len(self.waypoints) - 2)

        t_start = self.t_points[idx]
        t_end = self.t_points[idx + 1]
        dt_seg = max(1e-6, t_end - t_start)
        alpha = (t - t_start) / dt_seg

        p_d = (1.0 - alpha) * self.waypoints[idx] + alpha * self.waypoints[idx + 1]
        v_d = (self.waypoints[idx + 1] - self.waypoints[idx]) / dt_seg
        a_d = np.zeros(3)
        return p_d, v_d, a_d

class AnalyticEquationTrajectory:
    def __init__(self, eq_func, duration=25.0, sample_count=30):
        self.eq_func = eq_func
        self.total_duration = float(duration)
        t_samples = np.linspace(0.0, self.total_duration, sample_count)
        self.sampled_waypoints = np.array([self.eq_func(t) for t in t_samples], dtype=float)

    def evaluate(self, t):
        h = 1e-4
        p_d = np.asarray(self.eq_func(t), dtype=float)
        p_plus = np.asarray(self.eq_func(t + h), dtype=float)
        p_minus = np.asarray(self.eq_func(t - h), dtype=float)

        v_d = (p_plus - p_minus) / (2.0 * h)
        a_d = (p_plus - 2.0 * p_d + p_minus) / (h ** 2)
        return p_d, v_d, a_d

# ==============================================================================
# SECTION 11: FLIGHT CONTROL ALGORITHMS (SE(3) POSITION & ATTITUDE MAPPING)
# ==============================================================================
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
    z_B = F_des / thrust #finding in which direction the thrust is required

    x_C = np.array([
        np.cos(desired_yaw),
        np.sin(desired_yaw),
        0.0
    ]) #calculating the desired tilt of the quadrotor 

    y_B = np.cross(z_B, x_C) #finding an axis perpendicular to given inputs
    y_norm = np.linalg.norm(y_B) #to check if they are parallel

    if y_norm < 1e-6:

        x_C_fallback = np.array([
            -np.sin(desired_yaw),
            np.cos(desired_yaw),
            0.0
        ]) #this is the perpendicular axis to the original heading vector
        y_B = np.cross(z_B, x_C_fallback)
        y_norm = np.linalg.norm(y_B)

    y_B = y_B / y_norm

    x_B = np.cross(y_B, z_B)

    R = np.column_stack([x_B, y_B, z_B])

    q_des = rotation_matrix_to_quaternion(R)

    return thrust, q_des

# ==============================================================================
# SECTION 12: 3D SCENE RENDERING & VISUAL MARKERS
# ==============================================================================
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

# ==============================================================================
# SECTION 13: PRE-FLIGHT INITIALIZATION & HARDWARE SETUP
# ==============================================================================
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
    if TRAJECTORY_MODE == "CUSTOM_EQUATION":
        trajectory = AnalyticEquationTrajectory(custom_trajectory, duration=EQUATION_DURATION)
        WAYPOINTS_VIS = trajectory.sampled_waypoints
        traj_desc = "Custom Equation"
    elif TRAJECTORY_MODE == "PRESET_EQUATION":
        trajectory = AnalyticEquationTrajectory(lambda t: preset_trajectory(t, PRESET_SHAPE), duration=EQUATION_DURATION)
        WAYPOINTS_VIS = trajectory.sampled_waypoints
        traj_desc = f"Preset Equation ({PRESET_SHAPE})"
    else:  # "WAYPOINTS"
        if USE_BSPLINE:
            trajectory = BSplineWaypointTrajectory(WAYPOINTS, SEGMENT_DURATIONS)
            traj_desc = "Waypoints (B-Spline: ON)"
        else:
            trajectory = LinearWaypointTrajectory(WAYPOINTS, SEGMENT_DURATIONS)
            traj_desc = "Waypoints (B-Spline: OFF / Linear)"
        WAYPOINTS_VIS = WAYPOINTS

    SIM_TIME = trajectory.total_duration + 5.0
else:
    WAYPOINTS_VIS = WAYPOINTS
    traj_desc = "Attitude Mode"

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

    print(f"Trajectory mode : {traj_desc}")
    if TRAJECTORY_MODE == "WAYPOINTS":
        print("Waypoints:")
        for i, wp in enumerate(WAYPOINTS):
            print(f"  [{i}]  {wp}")

        print()

        print("Segment durations:")
        for i, d in enumerate(SEGMENT_DURATIONS):
            print(f"  Segment {i}→{i+1} : {d:.1f} s")
    else:
        print(f"Equation loop duration: {trajectory.total_duration:.1f} s")

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

print()

# ==============================================================================
# SECTION 14: INTERACTIVE VIEWER CONTROLS & EVENT HANDLERS
# ==============================================================================
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
    "infinite_loop": bool(INFINITE_LOOP),
    "recording_ab": False,
    "ab_start_time": 0.0,
    "ab_end_time": 0.0,
    "ab_segment_count": 0,
    "status_flash": "",
}
update_wind_vector(sim_state)

current_sim_time = [0.0]
ab_telemetry_buffer = []
saved_ab_segments = []
full_telemetry_buffer = []

def save_ab_csv(data_buffer, t_start, t_end, segment_num=1):
    if not data_buffer:
        print(f"\n[Telemetry Logger] Segment #{segment_num} buffer is empty; no points to save.")
        return None
    filename = f"telemetry_segment_{segment_num}_from_{t_start:.1f}s_to_{t_end:.1f}s.csv"
    csv_filename = os.path.join(PROJECT_ROOT, filename)
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time_s", "PosX", "PosY", "PosZ",
            "PosDesX", "PosDesY", "PosDesZ",
            "Roll_deg", "Pitch_deg", "Yaw_deg",
            "qW", "qX", "qY", "qZ",
            "Thrust_N",
            "Mx_Nm", "My_Nm", "Mz_Nm"
        ])
        for row in data_buffer:
            writer.writerow(row)
    saved_ab_segments.append((filename, t_start, t_end, len(data_buffer)))
    print(f"\n[Telemetry Logger] Segment #{segment_num} ({t_start:.2f}s to {t_end:.2f}s) exported to:\n  {filename} ({len(data_buffer)} data points)")
    return csv_filename

def key_callback(keycode):
    # 32 = SPACE, 80/112 = P/p (pause), 83/115 = S/s (start/play), 87/119 = W/w (wind)
    # 67/99 = C/c or 86/118 = V/v (camera mode cycle: 1:Orbit -> 2:Chase -> 3:FPV)
    # 61 / 265 = '+' / UP (increase force), 45 / 264 = '-' / DOWN (decrease force)
    # 93 / 262 = ']' / RIGHT (rotate CCW), 91 / 263 = '[' / LEFT (rotate CW)
    # 65/97 = A/a (mark Point A), 66/98 = B/b (mark Point B), 76/108 = L/l (toggle loop)
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
    elif keycode in (65, 97):  # 'A' or 'a' -> Mark Point A (Start recording)
        sim_state["recording_ab"] = True
        sim_state["ab_start_time"] = current_sim_time[0]
        next_seg = sim_state["ab_segment_count"] + 1
        sim_state["status_flash"] = f"[REC A #{next_seg}] Started @ {current_sim_time[0]:.1f}s"
        sim_state["ui_dirty"] = True
        ab_telemetry_buffer.clear()
        print(f"\n[Telemetry Logger] Point A MARKED for Segment #{next_seg} at t = {current_sim_time[0]:.2f} s. Logging STARTED...")
    elif keycode in (66, 98):  # 'B' or 'b' -> Mark Point B (Stop and save)
        if sim_state["recording_ab"]:
            sim_state["recording_ab"] = False
            sim_state["ab_end_time"] = current_sim_time[0]
            sim_state["ab_segment_count"] += 1
            seg_num = sim_state["ab_segment_count"]
            t_start = sim_state["ab_start_time"]
            t_end = sim_state["ab_end_time"]
            pts_count = len(ab_telemetry_buffer)
            save_ab_csv(ab_telemetry_buffer, t_start, t_end, segment_num=seg_num)
            ab_telemetry_buffer.clear()
            sim_state["status_flash"] = f"[SAVED B #{seg_num}] ({pts_count} pts, {t_start:.1f}s-{t_end:.1f}s)"
            sim_state["ui_dirty"] = True
        else:
            print("\n[Telemetry Logger] Point A has not been marked yet. Press [A] first to begin logging a segment.")
    elif keycode in (76, 108):  # 'L' or 'l' -> Toggle Infinite Loop
        sim_state["infinite_loop"] = not sim_state["infinite_loop"]
        sim_state["ui_dirty"] = True
        loop_str = "ENABLED (Infinite)" if sim_state["infinite_loop"] else "DISABLED (Finite)"
        print(f"\n[Loop Mode] Infinite Trajectory Looping: {loop_str}")

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
print("  [A]          : Mark Point A (START selective segment CSV logging)")
print("  [B]          : Mark Point B (STOP & export segment CSV — can repeat for multiple segments)")
print("  [L]          : Toggle Infinite Loop ON / OFF")
print("  Close the viewer window to finish and view plots.")
print()

# ==============================================================================
# SECTION 15: MAIN SIMULATION & FLIGHT CONTROL LOOP (500 Hz Physics)
# ==============================================================================
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
    while viewer.is_running() and (sim_state["infinite_loop"] or step < steps):

        cam_label = CAM_LABELS[sim_state["camera_mode"]]

        if sim_state["paused"]:
            if sim_state["ui_dirty"]:
                wind_str = f" | [WIND: {sim_state['wind_strength_n']:.1f}N @ {sim_state['wind_direction_deg']:.0f}°]" if sim_state["wind_active"] else ""
                loop_str = " | [LOOP: ∞]" if sim_state["infinite_loop"] else ""
                rec_str = f" | [REC A #{sim_state['ab_segment_count'] + 1} ({sim_state['ab_start_time']:.1f}s...)]" if sim_state["recording_ab"] else ""
                flash_str = f" | {sim_state['status_flash']}" if sim_state.get("status_flash") else ""
                viewer.set_texts([(
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    0,
                    "STATUS",
                    f"[PAUSED] | CAM: {cam_label}{wind_str}{loop_str}{rec_str}{flash_str} (SPACE: Play, V: Cam, W: Wind, A: Rec A, B: Save B, L: Loop)"
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
            loop_str = " | [LOOP: ∞]" if sim_state["infinite_loop"] else ""
            rec_str = f" | [REC A #{sim_state['ab_segment_count'] + 1} ({sim_state['ab_start_time']:.1f}s...)]" if sim_state["recording_ab"] else ""
            flash_str = f" | {sim_state['status_flash']}" if sim_state.get("status_flash") else ""
            viewer.set_texts([(
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                0,
                "STATUS",
                f"[RUNNING] | CAM: {cam_label}{wind_str}{loop_str}{rec_str}{flash_str} (SPACE: Pause, V: Cam, W: Wind, A: Rec A, B: Save B, L: Loop)"
            )])
            sim_state["ui_dirty"] = False
            print(f"\n[Viewer] Simulation RUNNING | CAM: {cam_label}{wind_str}{loop_str}{rec_str}")

        t = step * DT
        current_sim_time[0] = t

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

            if sim_state["infinite_loop"]:
                T_period = max(1.0, trajectory.total_duration)
                t_eval = t % T_period
            else:
                t_eval = t

            p_d, v_d, a_d = trajectory.evaluate(t_eval)

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
                WAYPOINTS_VIS,
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

        # Telemetry logging (at LOG_RATE_HZ, default 50 Hz -> every 10 steps at 500 Hz)
        log_interval = max(1, int(1.0 / (DT * LOG_RATE_HZ)))
        if step % log_interval == 0:
            log_row = [
                round(t, 4),
                round(pos[0], 5), round(pos[1], 5), round(pos[2], 5),
                round(p_d[0], 5), round(p_d[1], 5), round(p_d[2], 5),
                round(np.rad2deg(rpy[0]), 3), round(np.rad2deg(rpy[1]), 3), round(np.rad2deg(rpy[2]), 3),
                round(q_current[0], 5), round(q_current[1], 5), round(q_current[2], 5), round(q_current[3], 5),
                round(thrust_cmd, 3),
                round(torque[0], 6), round(torque[1], 6), round(torque[2], 6)
            ]
            if sim_state["recording_ab"]:
                ab_telemetry_buffer.append(log_row)
            if LOG_MODE in ("FULL", "BOTH"):
                full_telemetry_buffer.append(log_row)

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

# ==============================================================================
# SECTION 16: POST-FLIGHT ANALYSIS, CSV LOG EXPORT & 3D PLOTTING
# ==============================================================================
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

if sim_state["recording_ab"] and ab_telemetry_buffer:
    sim_state["ab_segment_count"] += 1
    seg_num = sim_state["ab_segment_count"]
    save_ab_csv(ab_telemetry_buffer, sim_state["ab_start_time"], current_sim_time[0], segment_num=seg_num)
    ab_telemetry_buffer.clear()

if saved_ab_segments:
    print(f"\n[Telemetry Logger] Exported {len(saved_ab_segments)} A->B segment(s):")
    for fname, t0, t1, count in saved_ab_segments:
        print(f"  • {fname}  ({t0:.2f}s to {t1:.2f}s, {count} pts)")

if (LOG_MODE in ("FULL", "BOTH") or SAVE_CSV) and full_telemetry_buffer:
    import csv
    csv_filename = os.path.join(PROJECT_ROOT, "simulation_data.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([
            "Time_s", "PosX", "PosY", "PosZ", 
            "PosDesX", "PosDesY", "PosDesZ",
            "Roll_deg", "Pitch_deg", "Yaw_deg", 
            "qW", "qX", "qY", "qZ",
            "Thrust_N",
            "Mx_Nm", "My_Nm", "Mz_Nm"
        ])

        for row in full_telemetry_buffer:
            writer.writerow(row)

    print(f"\nFull simulation data saved to {csv_filename} ({len(full_telemetry_buffer)} points logged at {LOG_RATE_HZ} Hz).")
elif LOG_MODE == "AB_ONLY":
    print(f"\n[Logging Mode: AB_ONLY] Zero background logging overhead active.")

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

    wp_arr = np.array(WAYPOINTS_VIS)

    ax3d.scatter(
        wp_arr[:, 0],
        wp_arr[:, 1],
        wp_arr[:, 2],
        c="red",
        s=50,
        marker="o",
        label="Waypoints" if TRAJECTORY_MODE == "WAYPOINTS" else "Sampled Equation Pts"
    )

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title(f"3D Trajectory — {traj_desc}")
    ax3d.legend()

plt.show()
