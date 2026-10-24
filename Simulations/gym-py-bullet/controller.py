"""
Quadrotor Controllers: Quaternion P2 Attitude Controller, Position Controller,
and Motor Mixer.

CONVENTIONS & MATHEMATICAL FOUNDATIONS:
----------------------------------------
1. Quaternion Format:
   - Internal / MuJoCo format: [w, x, y, z] (scalar-first: w is real, [x, y, z] is vector).
   - PyBullet API format:     [x, y, z, w] (scalar-last: w is at index 3).
   - Explicit conversion helpers: `pybullet_to_mujoco_quat` and `mujoco_to_pybullet_quat`.

2. Quaternion Attitude Error & Multiplication Order:
   - q_meas represents the rotation from body frame to world frame (v_world = R(q_meas) * v_body).
   - q_ref represents the desired rotation from body frame to world frame.
   - The attitude error from current body orientation to reference orientation in BODY coordinates is:
         q_err_body = q_meas* ⊗ q_ref
     Because q_meas ⊗ q_err_body = q_ref.
   - When q_err_body[0] < 0, we negate q_err_body = -q_err_body (shortest geodesic path on S3).
   - The vector part axis_error = q_err_body[1:4] points along the body-fixed rotation axis.

3. Feedback Sign:
   - Because q_err_body = q_meas* ⊗ q_ref represents the required body rotation to reach q_ref,
     a POSITIVE torque along axis_error drives the drone toward the reference:
         tau_body = +Pq * axis_error - Pw * omega_body
     where -Pw * omega_body provides angular rate damping.

4. Motor Mixing (+ Configuration):
   - Rotor 1 (Front, +X): Pitch down (-tau_y), CCW rotor (-km reaction)
   - Rotor 2 (Rear,  -X): Pitch up   (+tau_y), CCW rotor (-km reaction)
   - Rotor 3 (Left,  +Y): Roll right (+tau_x), CW rotor  (+km reaction)
   - Rotor 4 (Right, -Y): Roll left  (-tau_x), CW rotor  (+km reaction)
"""

import numpy as np


# ==============================================================================
# QUATERNION & ROTATION UTILITIES
# ==============================================================================

def pybullet_to_mujoco_quat(q_pb):
    """Convert PyBullet quaternion [x, y, z, w] to scalar-first [w, x, y, z]."""
    x, y, z, w = q_pb
    return np.array([w, x, y, z], dtype=float)


def mujoco_to_pybullet_quat(q_mj):
    """Convert scalar-first [w, x, y, z] to PyBullet quaternion [x, y, z, w]."""
    w, x, y, z = q_mj
    return np.array([x, y, z, w], dtype=float)


def normalize_quaternion(q):
    """Normalize quaternion [w, x, y, z]. Returns unit quaternion."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_conjugate(q):
    """Conjugate of quaternion [w, x, y, z] -> [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_multiply(q1, q2):
    """
    Hamilton quaternion product q = q1 ⊗ q2 for [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quaternion_to_rotation_matrix(q):
    """
    Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix R (Body -> World).
    """
    q = normalize_quaternion(q)
    w, x, y, z = q

    return np.array([
        [1.0 - 2.0*(y*y + z*z),  2.0*(x*y - z*w),        2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),        2.0*(y*z + x*w),        1.0 - 2.0*(x*x + y*y)]
    ], dtype=float)


def rotation_matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix R to unit quaternion [w, x, y, z] with numerical stability.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return normalize_quaternion(np.array([w, x, y, z]))


def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw (radians, ZYX intrinsic convention) to quaternion [w, x, y, z].
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return normalize_quaternion(np.array([qw, qx, qy, qz]))


def quaternion_to_rpy(q):
    """
    Convert unit quaternion [w, x, y, z] to roll, pitch, yaw (radians).
    """
    q = normalize_quaternion(q)
    qw, qx, qy, qz = q

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=float)


# ==============================================================================
# P2 QUATERNION ATTITUDE CONTROLLER
# ==============================================================================

class QuaternionP2Controller:
    """
    Quaternion-based proportional-derivative (P2) attitude controller.
    Computes body torques from attitude quaternion error and angular rate damping.
    """

    def __init__(self, Pq=16.0, Pw=1.8, torque_limit=0.6):
        self.Pq = float(Pq)
        self.Pw = float(Pw)
        self.torque_limit = float(torque_limit)

    def compute_error(self, q_ref, q_measured, shortest_path=True):
        """
        Calculate attitude error in the BODY frame.

        Args:
            q_ref: Desired attitude quaternion [w, x, y, z].
            q_measured: Current attitude quaternion [w, x, y, z].
            shortest_path: If True, choose shortest rotation on S3 (q_err[0] >= 0).

        Returns:
            q_err: Error quaternion [w, x, y, z].
            axis_error: 3D vector part of error quaternion [x, y, z].
        """
        q_ref = normalize_quaternion(q_ref)
        q_meas = normalize_quaternion(q_measured)

        # Body-frame error: rotation from current body to reference body
        # q_err = q_meas* ⊗ q_ref
        q_err = quaternion_multiply(quaternion_conjugate(q_meas), q_ref)

        if shortest_path and q_err[0] < 0.0:
            q_err = -q_err

        axis_error = q_err[1:4]
        return q_err, axis_error

    def compute_torque(self, q_ref, q_measured, omega_body, shortest_path=True):
        """
        Calculate control torque vector in body coordinates.

        Args:
            q_ref: Desired attitude [w, x, y, z].
            q_measured: Current attitude [w, x, y, z].
            omega_body: Current angular velocity in BODY coordinates [wx, wy, wz] (rad/s).
            shortest_path: Whether to use shortest geodesic rotation.

        Returns:
            torque: Commanded 3D torque in body frame [tau_x, tau_y, tau_z] (N*m).
            q_err: Quaternion error [w, x, y, z].
        """
        q_err, axis_error = self.compute_error(q_ref, q_measured, shortest_path=shortest_path)
        omega_body = np.asarray(omega_body, dtype=float)

        # P2 Control Law:
        # +Pq * axis_error drives body toward q_ref
        # -Pw * omega_body provides derivative damping
        torque = self.Pq * axis_error - self.Pw * omega_body

        # Saturation clamp
        torque = np.clip(torque, -self.torque_limit, self.torque_limit)

        return torque, q_err


# ==============================================================================
# POSITION CONTROLLER & DIFFERENTIAL FLATNESS MAPPING
# ==============================================================================

class PositionController:
    """
    Calculates commanded 3D acceleration from tracking errors, then derives
    total collective thrust magnitude and desired attitude quaternion.
    """

    def __init__(self, Kp, Kd, Ki=None, max_accel=10.0, mass=1.0, gravity=9.81):
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)
        self.Ki = np.zeros(3) if Ki is None else np.asarray(Ki, dtype=float)
        self.max_accel = float(max_accel)
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.integral_error = np.zeros(3)

    def reset(self):
        self.integral_error = np.zeros(3)

    def compute_control(self, pos, vel, pos_d, vel_d, accel_d, dt, desired_yaw_rad=0.0):
        """
        Compute total thrust and desired attitude quaternion from position error.

        Returns:
            thrust_cmd: Total required thrust magnitude (N).
            q_des: Desired attitude quaternion [w, x, y, z].
            accel_cmd: Commanded Cartesian acceleration (m/s^2).
        """
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        pos_d = np.asarray(pos_d, dtype=float)
        vel_d = np.asarray(vel_d, dtype=float)
        accel_d = np.asarray(accel_d, dtype=float)

        pos_error = pos_d - pos
        vel_error = vel_d - vel

        # Integrate with anti-windup clamp
        self.integral_error += pos_error * dt
        self.integral_error = np.clip(self.integral_error, -2.0, 2.0)

        # PID + Feedforward acceleration command
        accel_cmd = (
            accel_d
            + self.Kp * pos_error
            + self.Kd * vel_error
            + self.Ki * self.integral_error
        )

        # Acceleration saturation
        accel_mag = np.linalg.norm(accel_cmd)
        if accel_mag > self.max_accel:
            accel_cmd = accel_cmd * (self.max_accel / accel_mag)

        # Required force in world coordinates: F_des = m * (a_cmd + g * z_hat)
        g_vec = np.array([0.0, 0.0, self.gravity])
        F_des = self.mass * (accel_cmd + g_vec)

        # Safeguard: If commanded vertical force is non-positive, stay upright
        if F_des[2] <= 0.05:
            q_upright = np.array([
                np.cos(desired_yaw_rad * 0.5),
                0.0,
                0.0,
                np.sin(desired_yaw_rad * 0.5)
            ])
            return 0.0, q_upright, accel_cmd

        thrust_cmd = float(np.linalg.norm(F_des))

        # Desired body Z axis in world frame
        z_B = F_des / thrust_cmd

        # Desired heading vector in horizontal plane
        x_C = np.array([np.cos(desired_yaw_rad), np.sin(desired_yaw_rad), 0.0])

        # Orthogonal body Y axis: y_B = (z_B x x_C) / ||z_B x x_C||
        y_B = np.cross(z_B, x_C)
        y_norm = np.linalg.norm(y_B)
        if y_norm < 1e-6:
            # Fallback when z_B is collinear with x_C
            x_C_fallback = np.array([-np.sin(desired_yaw_rad), np.cos(desired_yaw_rad), 0.0])
            y_B = np.cross(z_B, x_C_fallback)
            y_norm = np.linalg.norm(y_B)

        y_B = y_B / y_norm
        # Orthogonal body X axis completes right-handed orthonormal triad
        x_B = np.cross(y_B, z_B)

        # Rotation matrix R = [x_B, y_B, z_B] (Body -> World)
        R_des = np.column_stack([x_B, y_B, z_B])
        q_des = rotation_matrix_to_quaternion(R_des)

        return thrust_cmd, q_des, accel_cmd

    def compute_attitude_hold_control(
        self,
        pos,
        vel,
        current_rpy,
        target_rpy_rad,
        hold_pos,
        safety_radius=2.0,
        dt=0.002
    ):
        """
        Cascaded position-hold controller for the attitude experiment.
        
        Balances commanded attitude targets [roll, pitch, yaw] with position holding:
          1. Altitude Control: Computes collective thrust to maintain hold_pos[2],
             compensating for thrust vector tilt (1 / cos(tilt)).
          2. Horizontal Drift Compensation: Calculates corrective tilt angles needed
             to counteract lateral drift induced by roll/pitch tilt forces.
          3. Explicit Command Blending: Uses a continuous distance-based weight w in [0, 1]
             with velocity anticipation and smoothstep transitions so commanded attitude
             is tested when near hold_pos, while restoring tilt smoothly engages to keep
             the drone tightly bounded near hold_pos.
          4. Safety Verification: Flags whether the drone remains within the safe arena boundary.

        Returns:
            thrust_cmd: Collective thrust magnitude in Newtons.
            q_des: Desired attitude quaternion [w, x, y, z].
            rpy_des: Desired roll, pitch, yaw in radians.
            is_safe: Boolean indicating whether drone is within safe boundary.
            pos_error: 3D position error vector (hold_pos - pos).
        """
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        hold_pos = np.asarray(hold_pos, dtype=float)
        target_rpy_rad = np.asarray(target_rpy_rad, dtype=float)
        phi_target, theta_target, psi_target = target_rpy_rad

        # 1. Position error vector
        pos_error = hold_pos - pos

        # 2. Vertical Altitude Control (Z-axis)
        z_err = pos_error[2]
        vz_err = -vel[2]
        # Tilt angle magnitude to prevent thrust runaway at high tilt
        current_tilt = np.sqrt(current_rpy[0]**2 + current_rpy[1]**2)
        tilt_cos = float(np.clip(np.cos(current_tilt), 0.5, 1.0))

        # Commanded vertical acceleration
        az_cmd = self.Kp[2] * z_err + self.Kd[2] * vz_err
        F_z = self.mass * (self.gravity + az_cmd)
        thrust_cmd = float(np.clip(F_z / tilt_cos, 0.0, 4.0 * 10.0))

        # 3. Horizontal Position Compensation (X and Y axes)
        e_x = pos_error[0]
        e_y = pos_error[1]
        dist_xy = float(np.hypot(e_x, e_y))

        # Horizontal corrective accelerations: a_pos = Kp * e - Kd * v
        ax_pos = self.Kp[0] * e_x - self.Kd[0] * vel[0]
        ay_pos = self.Kp[1] * e_y - self.Kd[1] * vel[1]

        # Convert corrective accelerations to corrective tilt angles
        # +ax accelerates in +x -> positive pitch theta
        # +ay accelerates in +y -> negative roll phi
        max_tilt_corr = 0.50  # ~28.6 degrees max corrective tilt
        theta_pos = float(np.clip(ax_pos / self.gravity, -max_tilt_corr, max_tilt_corr))
        phi_pos = float(np.clip(-ay_pos / self.gravity, -max_tilt_corr, max_tilt_corr))

        # 4. Explicit Command Resolution & Boundary Blending
        # Incorporate velocity anticipation in radial direction to brake before overshoot
        v_radial = (vel[0] * (-e_x) + vel[1] * (-e_y)) / (dist_xy + 1e-6)
        effective_dist = dist_xy + 0.45 * max(0.0, v_radial)

        # Smoothstep blending between inner (0.2m) and outer (1.1m) radii
        r_inner = 0.20
        r_outer = min(1.10, safety_radius * 0.70)
        if effective_dist <= r_inner:
            w = 0.0
        elif effective_dist >= r_outer:
            w = 1.0
        else:
            s = (effective_dist - r_inner) / (r_outer - r_inner)
            w = s * s * (3.0 - 2.0 * s)

        # Blended desired roll and pitch
        phi_des = (1.0 - w) * phi_target + w * phi_pos
        theta_des = (1.0 - w) * theta_target + w * theta_pos
        # Yaw has zero translation coupling, so it tracks target yaw 100%
        psi_des = psi_target

        rpy_des = np.array([phi_des, theta_des, psi_des], dtype=float)
        q_des = rpy_to_quaternion(phi_des, theta_des, psi_des)

        # 5. Safety check
        is_safe = dist_xy <= safety_radius

        return thrust_cmd, q_des, rpy_des, is_safe, pos_error


# ==============================================================================
# MOTOR MIXER (+ CONFIGURATION)
# ==============================================================================

class MotorMixer:
    """
    Distributes total thrust and 3-axis body torques to 4 individual rotor thrusts.
    Configuration: "+" configuration with arm length L.
      Rotor 1: (+L, 0) - Front
      Rotor 2: (-L, 0) - Rear
      Rotor 3: (0, +L) - Left
      Rotor 4: (0, -L) - Right
    """

    def __init__(self, arm_length=0.45, km=0.016, min_thrust=0.0, max_thrust=10.0):
        self.L = float(arm_length)
        self.km = float(km)
        self.min_thrust = float(min_thrust)
        self.max_thrust = float(max_thrust)

    def mix(self, total_thrust, torque_cmd):
        """
        Convert total thrust and 3D body torque into 4 motor thrust forces.

        Args:
            total_thrust: Commanded collective thrust (N).
            torque_cmd: Commanded body torques [tau_x, tau_y, tau_z] (N*m).

        Returns:
            motor_forces: Array of 4 rotor thrusts [f1, f2, f3, f4] (N).
            actual_thrust: Resultant total thrust after motor saturation (N).
            actual_torques: Resultant body torques after saturation [tx, ty, tz] (N*m).
        """
        tx, ty, tz = torque_cmd

        # Exact linear motor allocation equations:
        # T   = f1 + f2 + f3 + f4
        # tau_x = L * (f3 - f4)
        # tau_y = L * (f2 - f1)
        # tau_z = km * (-f1 - f2 + f3 + f4)
        f_base = total_thrust / 4.0
        delta_pitch = ty / (2.0 * self.L)
        delta_roll  = tx / (2.0 * self.L)
        delta_yaw   = tz / (4.0 * self.km)

        f1 = f_base - delta_pitch - delta_yaw  # Rotor 1 (Front)
        f2 = f_base + delta_pitch - delta_yaw  # Rotor 2 (Rear)
        f3 = f_base + delta_roll  + delta_yaw  # Rotor 3 (Left)
        f4 = f_base - delta_roll  + delta_yaw  # Rotor 4 (Right)

        # Apply motor thrust saturation limits
        motor_forces = np.clip([f1, f2, f3, f4], self.min_thrust, self.max_thrust)

        # Compute actual forces and torques produced by the saturated motors
        actual_thrust = float(np.sum(motor_forces))
        actual_tx = self.L * (motor_forces[2] - motor_forces[3])
        actual_ty = self.L * (motor_forces[1] - motor_forces[0])
        actual_tz = self.km * (-motor_forces[0] - motor_forces[1] + motor_forces[2] + motor_forces[3])
        actual_torques = np.array([actual_tx, actual_ty, actual_tz], dtype=float)

        return motor_forces, actual_thrust, actual_torques
