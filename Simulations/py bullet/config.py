"""
Simulation Configuration and Waypoint Settings.

You can edit waypoints, flight duration, and experiment mode directly below.
"""

import numpy as np

# ==============================================================================
# 1. EXPERIMENT MODE SELECTION
# ==============================================================================
# Select one of:
#   - "trajectory" : Drone follows the smooth B-spline path through WAYPOINTS (Default)
#   - "attitude"   : Drone tests roll, pitch, and yaw tracking in place
MODE = "trajectory"


# ==============================================================================
# 2. EDIT YOUR WAYPOINTS HERE (TRAJECTORY MODE)
# ==============================================================================
# Format: [x, y, z] in world coordinates (meters).
# Z is altitude measured upward from the ground plane (Z=0).
# You can freely add, remove, or modify waypoints in this list!
WAYPOINTS = [
    [0.0, 0.0, 1.0],   # WP1: Takeoff / initial hover
    [2.0, 0.0, 2.0],   # WP2: Forward climb
    [4.0, 2.0, 3.0],   # WP3: High curve right
    [6.0, 4.0, 2.5],   # WP4: Banking descent
    [8.0, 2.0, 2.0],   # WP5: Return curve
    [6.0, 0.0, 1.5],   # WP6: Approach
    [3.0, -1.0, 1.2],  # WP7: Final leg
    [0.0, 0.0, 1.0],   # WP8: Return to home / hover
]

# Total flight duration for trajectory following (seconds)
FLIGHT_DURATION = 16.0

# Desired drone heading / yaw angle along the trajectory (degrees)
DESIRED_YAW_DEG = 0.0


# ==============================================================================
# 3. ATTITUDE EXPERIMENT CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ==============================================================================
# Commanded orientation targets (in degrees):
# Setting an angle to 0.0 means the drone maintains zero rotation about that axis.
ATTITUDE_ROLL_DEG = 90.0       # Target roll angle (degrees, rotation about body X-axis)
ATTITUDE_PITCH_DEG = 0.0      # Target pitch angle (degrees, rotation about body Y-axis)
ATTITUDE_YAW_DEG = 0.0        # Target yaw angle (degrees, rotation about body Z-axis)

# Fixed position for attitude testing, in metres [x, y, z]
ATTITUDE_HOLD_POSITION = [0.0, 0.0, 1.0]

# Duration of attitude test run in seconds
ATTITUDE_TEST_DURATION = 8.0

# Safety boundary radius (metres): maximum allowable horizontal drift
ATTITUDE_SAFETY_RADIUS = 2.0


# ==============================================================================
# 4. QUADROTOR PHYSICAL PARAMETERS (MATCHING MUJOCO MODEL)
# ==============================================================================
# Total drone mass (kg)
MASS = 1.0

# Diagonal inertia tensor [Ixx, Iyy, Izz] (kg*m^2)
# Exactly matching MuJoCo diaginertia="0.00065 0.00065 0.0012"
INERTIA = np.array([0.00065, 0.00065, 0.0012], dtype=float)

# Arm length L (m) from center of mass to rotor center
ARM_LENGTH = 0.45

# Rotor configuration: "+" configuration (front=+X, rear=-X, left=+Y, right=-Y)
# Rotor 1: (+L, 0, z)  - Pitch down torque
# Rotor 2: (-L, 0, z)  - Pitch up torque
# Rotor 3: (0, +L, z)  - Roll right torque
# Rotor 4: (0, -L, z)  - Roll left torque
ROTOR_POSITIONS = np.array([
    [ ARM_LENGTH,  0.0,         0.03],  # Rotor 1 (Front)
    [-ARM_LENGTH,  0.0,         0.03],  # Rotor 2 (Rear)
    [ 0.0,         ARM_LENGTH,  0.03],  # Rotor 3 (Left)
    [ 0.0,        -ARM_LENGTH,  0.03],  # Rotor 4 (Right)
], dtype=float)

# Rotor reaction torque coefficient (tau_z = KM * thrust)
KM = 0.016

# Rotor thrust limits per motor (N)
MIN_MOTOR_THRUST = 0.0
MAX_MOTOR_THRUST = 10.0  # Total max thrust = 40 N (4g max thrust capability)

# Hover thrust per motor = (MASS * 9.81) / 4 ~ 2.4525 N
GRAVITY = 9.81
HOVER_THRUST = MASS * GRAVITY

# Total thrust limits
MIN_TOTAL_THRUST = 0.0
MAX_TOTAL_THRUST = 4 * MAX_MOTOR_THRUST


# ==============================================================================
# 5. CONTROLLER GAINS & LIMITS
# ==============================================================================
# Outer-loop Position Controller (PID / PD + Feedforward)
Kp_POS = np.array([12.0, 12.0, 15.0], dtype=float)
Kd_POS = np.array([6.5, 6.5, 7.5], dtype=float)
Ki_POS = np.array([0.1, 0.1, 0.2], dtype=float)
MAX_ACCEL = 10.0  # Maximum commanded linear acceleration (m/s^2)

# Inner-loop Quaternion P2 Attitude Controller
# Law: tau = +Pq * axis_error - Pw * omega
# (Positive feedback sign matches body error rotation from current to desired)
Pq_ATT = 16.0     # Proportional attitude quaternion gain
Pw_ATT = 1.8      # Angular rate damping gain
TORQUE_LIMIT = 0.6  # Maximum torque command (N*m)


# ==============================================================================
# 6. SIMULATION & VISUALIZATION SETTINGS
# ==============================================================================
DT = 0.002            # Simulation timestep (500 Hz, matching MuJoCo DT=0.002)
REAL_TIME_SYNC = True # Sleep to maintain real-time 1x speed in GUI visualizer

# Camera view preset (designed to showcase the entire arena & trajectory)
CAMERA_DISTANCE = 8.5
CAMERA_YAW = 45.0
CAMERA_PITCH = -30.0
CAMERA_TARGET = [3.5, 1.5, 1.0]

# Waypoint Marker Visuals
WP_RADIUS = 0.14
WP_COLOR_START = [0.1, 0.9, 0.2, 0.9]    # Vivid Green for WP1
WP_COLOR_END   = [0.95, 0.15, 0.15, 0.9] # Vivid Red for final WP
WP_COLOR_MID   = [0.15, 0.65, 1.0, 0.85] # Cyan for intermediate WPs
LINE_COLOR_DESIRED = [1.0, 0.75, 0.1]    # Gold/Yellow for reference path
LINE_COLOR_ACTUAL  = [0.0, 0.9, 0.4]     # Bright Green for drone trail
LINE_WIDTH = 2.5
