"""
==============================================================================
PYBULLET QUADROTOR SIMULATION WITH P2 ATTITUDE CONTROL & B-SPLINE TRACKING
==============================================================================

This script implements:
  1. Quadrotor rigid-body physics in PyBullet with gravity.
  2. Quaternion-based P2 attitude controller derived from the MuJoCo model.
  3. Cascaded position and attitude control architecture.
  4. 4-rotor motor mixer with realistic thrust limits and reaction torques.
  5. Two supported experiment modes:
     - "trajectory": Smooth C2-continuous B-spline waypoint tracking (Default).
                     Opens an interactive 3D Matplotlib window to rotate the plot.
     - "attitude"  : Fixed-position attitude tracking hovering 1 unit above ground.
                     Tests roll (e.g. 90°), pitch, and yaw without translation.
  6. Interactive PyBullet camera: Click and drag anywhere with the mouse or use
     keyboard arrow keys to orbit/rotate the view; scroll wheel to zoom.
  7. Telemetry logging and automatic plot generation in results/.

Execution:
  .\\run.ps1                   (Trajectory mode, default)
  .\\run.ps1 -mode attitude     (Attitude mode)
"""

import os
import sys
import time
import argparse
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

# Local module imports
import config
from trajectory import BSplineWaypointTrajectory
from controller import (
    QuaternionP2Controller,
    PositionController,
    MotorMixer,
    pybullet_to_mujoco_quat,
    quaternion_to_rpy,
    rpy_to_quaternion,
    quaternion_to_rotation_matrix,
)


# ==============================================================================
# ARENA BUILDER & VISUAL MARKERS
# ==============================================================================

def build_testing_arena():
    """
    Constructs a dedicated flight arena distinct from the MuJoCo urban scene.
    Includes:
      - Dark reflective ground plane
      - Circular takeoff/landing pad (helipad) with high-contrast markings
      - Measurement grid lines on the flight floor
      - Arena boundary safety pylons at the perimeter
    """
    # 1. Base ground plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.14, 0.16, 0.18, 1.0])

    # 2. Helipad Takeoff & Landing Platform at (0, 0, 0)
    pad_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=1.3,
        length=0.015,
        rgbaColor=[0.18, 0.22, 0.28, 1.0]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=pad_visual, basePosition=[0, 0, 0.008])

    # Outer yellow perimeter ring
    ring_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=1.1,
        length=0.02,
        rgbaColor=[0.95, 0.75, 0.1, 1.0]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=ring_visual, basePosition=[0, 0, 0.01])

    # Inner platform core
    inner_disc = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.95,
        length=0.025,
        rgbaColor=[0.15, 0.18, 0.22, 1.0]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=inner_disc, basePosition=[0, 0, 0.012])

    # Helipad "H" Marking (visual box bars)
    bar_v1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.05, 0.005], rgbaColor=[1.0, 1.0, 1.0, 0.95])
    bar_v2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.05, 0.005], rgbaColor=[1.0, 1.0, 1.0, 0.95])
    bar_h  = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.20, 0.006], rgbaColor=[1.0, 1.0, 1.0, 0.95])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=bar_v1, basePosition=[0, -0.20, 0.026])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=bar_v2, basePosition=[0,  0.20, 0.026])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=bar_h,  basePosition=[0,  0.00, 0.027])

    # 3. Ground Measurement Grid Markings
    grid_color = [0.28, 0.32, 0.38]
    grid_extent = 10.0
    step = 2.0
    for x in np.arange(-2.0, grid_extent + step, step):
        p.addUserDebugLine([x, -4.0, 0.002], [x, 6.0, 0.002], lineColorRGB=grid_color, lineWidth=1.0)
    for y in np.arange(-4.0, 6.0 + step, step):
        p.addUserDebugLine([-2.0, y, 0.002], [grid_extent, y, 0.002], lineColorRGB=grid_color, lineWidth=1.0)

    # 4. Arena Boundary Pylons (Four Corners)
    corners = [[-2.5, -4.5], [10.5, -4.5], [10.5, 6.5], [-2.5, 6.5]]
    for cx, cy in corners:
        pylon_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.08,
            length=1.5,
            rgbaColor=[0.85, 0.45, 0.1, 0.85]
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=pylon_vis, basePosition=[cx, cy, 0.75])


def create_waypoint_visuals(waypoints, trajectory=None):
    """
    Renders 3D waypoint markers and trajectory path lines in the PyBullet world.
    Markers are purely visual (no collision) and adapt dynamically to any waypoint list.
    """
    n_pts = len(waypoints)

    for i, wp in enumerate(waypoints):
        if i == 0:
            color = config.WP_COLOR_START  # Vivid Green
            label = f"WP{i+1} (Start)"
            textColor = [0.1, 1.0, 0.3]
        elif i == n_pts - 1:
            color = config.WP_COLOR_END    # Vivid Red
            label = f"WP{i+1} (Goal)"
            textColor = [1.0, 0.2, 0.2]
        else:
            color = config.WP_COLOR_MID    # Vivid Cyan
            label = f"WP{i+1}"
            textColor = [0.2, 0.8, 1.0]

        vis_sphere = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=config.WP_RADIUS,
            rgbaColor=color
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_sphere,
            basePosition=wp
        )

        p.addUserDebugText(
            text=label,
            textPosition=[wp[0], wp[1], wp[2] + 0.28],
            textColorRGB=textColor,
            textSize=1.3,
            lifeTime=0
        )

    # Draw continuous B-spline reference trajectory line
    if trajectory is not None:
        path_samples = trajectory.sample_path(num_samples=350)
        for j in range(len(path_samples) - 1):
            p.addUserDebugLine(
                path_samples[j],
                path_samples[j + 1],
                lineColorRGB=config.LINE_COLOR_DESIRED,
                lineWidth=config.LINE_WIDTH,
                lifeTime=0
            )


def create_attitude_arena_visuals(hold_pos):
    """
    Renders visual indicators for attitude mode:
      - 3D target hold sphere at hold_pos
      - Setpoint text label
    """
    vis_sphere = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.10,
        rgbaColor=[0.15, 0.85, 1.0, 0.75]
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis_sphere,
        basePosition=hold_pos
    )

    # Floating 3D label
    p.addUserDebugText(
        text=f"Hover Setpoint [{hold_pos[0]:.1f}, {hold_pos[1]:.1f}, {hold_pos[2]:.1f}]m",
        textPosition=[hold_pos[0], hold_pos[1], hold_pos[2] + 0.35],
        textColorRGB=[0.15, 0.9, 1.0],
        textSize=1.3,
        lifeTime=0
    )


# ==============================================================================
# MAIN SIMULATION RUNNER
# ==============================================================================

def run_simulation(flight_mode=None):
    """
    Initializes PyBullet with GUI enabled, runs the closed-loop flight simulation,
    logs all state variables, and produces verification plots.
    """
    mode = flight_mode if flight_mode is not None else config.MODE
    if mode not in ["trajectory", "attitude"]:
        raise ValueError(f"Invalid mode '{mode}'. Supported modes are: 'trajectory', 'attitude'.")

    dt = config.DT

    print("\n" + "=" * 65)
    print(" PYBULLET QUADROTOR SIMULATION WITH P2 ATTITUDE CONTROL")
    print("=" * 65)
    print(f"  Flight Mode          : {mode.upper()}")
    print(f"  Simulation Timestep  : {dt} s (500 Hz)")
    print(f"  Gravity Vector       : [0, 0, {-config.GRAVITY:.2f}] m/s^2")
    print(f"  Drone Mass           : {config.MASS:.2f} kg")
    print(f"  Drone Inertia Tensor : {config.INERTIA} kg*m^2")
    print(f"  Arm Length L         : {config.ARM_LENGTH:.2f} m")
    print(f"  P2 Attitude Gains    : Pq={config.Pq_ATT}, Pw={config.Pw_ATT}")
    print(f"  Torque Limit         : ±{config.TORQUE_LIMIT:.2f} Nm")

    if mode == "trajectory":
        waypoints = list(config.WAYPOINTS)
        total_flight_time = float(config.FLIGHT_DURATION)
        print(f"  Trajectory Duration  : {total_flight_time:.1f} s")
        print(f"  Number of Waypoints  : {len(waypoints)}")
        for idx, pt in enumerate(waypoints):
            print(f"    WP{idx+1}: {pt}")
    else:  # attitude mode
        waypoints = None
        print(f"  Hover Position       : {config.ATTITUDE_HOLD_POSITION} m (Hovering 1 unit above ground)")
        print(f"  Target Roll Angle    : {config.ATTITUDE_ROLL_DEG:+.1f} deg")
        print(f"  Target Pitch Angle   : {config.ATTITUDE_PITCH_DEG:+.1f} deg")
        print(f"  Target Yaw Angle     : {config.ATTITUDE_YAW_DEG:+.1f} deg")
        print(f"  Test Duration        : {config.ATTITUDE_TEST_DURATION:.1f} s")
    print("=" * 65)
    print("  CAMERA CONTROLS:")
    print("    - Click and drag mouse anywhere in the window to rotate the 3D world.")
    print("    - Use Arrow Keys [<- -> ^ v] to rotate camera.")
    print("    - Scroll mouse wheel to zoom in / out.")
    print("=" * 65 + "\n")

    # 1. Connect to PyBullet physics server (Always GUI)
    physics_client = p.connect(p.GUI)

    # Configure visualizer options
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)  # Disable object grabbing so click-drag rotates camera

    if mode == "attitude":
        cam_target = list(config.ATTITUDE_HOLD_POSITION)
        cam_dist = 3.2
        cam_yaw = 40.0
        cam_pitch = -15.0
    else:
        cam_target = list(config.CAMERA_TARGET)
        cam_dist = config.CAMERA_DISTANCE
        cam_yaw = config.CAMERA_YAW
        cam_pitch = config.CAMERA_PITCH

    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=cam_yaw,
        cameraPitch=cam_pitch,
        cameraTargetPosition=cam_target
    )

    p.setGravity(0, 0, -config.GRAVITY)
    p.setTimeStep(dt)

    # 2. Build Testing Arena and Visual Markers
    build_testing_arena()

    # 3D Visual Camera Guide floating in the world
    p.addUserDebugText(
        text="Controls: Drag Mouse or Arrow Keys to Orbit Camera | Scroll to Zoom",
        textPosition=[-1.8, 0.0, 3.2],
        textColorRGB=[1.0, 0.85, 0.2],
        textSize=1.1,
        lifeTime=0
    )

    if mode == "trajectory":
        trajectory = BSplineWaypointTrajectory(waypoints, total_duration=total_flight_time)
        sim_duration = trajectory.total_duration + 3.0  # Allow hover at goal
        create_waypoint_visuals(waypoints, trajectory=trajectory)
        init_pos = [waypoints[0][0], waypoints[0][1], 0.12]
        init_orn = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion [x, y, z, w]
    else:  # attitude mode
        trajectory = None
        sim_duration = float(config.ATTITUDE_TEST_DURATION)
        create_attitude_arena_visuals(config.ATTITUDE_HOLD_POSITION)
        # Position drone directly at ATTITUDE_HOLD_POSITION (1 unit above ground)
        init_pos = list(config.ATTITUDE_HOLD_POSITION)
        init_orn = [0.0, 0.0, 0.0, 1.0]

    # 3. Load Quadrotor URDF
    urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quadrotor.urdf")
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"quadrotor.urdf not found at {urdf_path}")

    drone_id = p.loadURDF(urdf_path, basePosition=init_pos, baseOrientation=init_orn)

    # Enable contact damping
    p.changeDynamics(drone_id, -1, linearDamping=0.01, angularDamping=0.01)

    # In attitude mode, create a spherical constraint at ATTITUDE_HOLD_POSITION.
    # This locks translation so the drone hovers exactly 1 unit above the ground without
    # moving horizontally or vertically, while leaving all 3 rotational DOFs (roll, pitch, yaw)
    # 100% free to be driven dynamically by the quaternion P2 attitude controller.
    if mode == "attitude":
        attitude_constraint_id = p.createConstraint(
            parentBodyUniqueId=drone_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=config.ATTITUDE_HOLD_POSITION
        )

    # 4. Initialize Controllers
    pos_controller = PositionController(
        Kp=config.Kp_POS,
        Kd=config.Kd_POS,
        Ki=config.Ki_POS,
        max_accel=config.MAX_ACCEL,
        mass=config.MASS,
        gravity=config.GRAVITY
    )

    p2_controller = QuaternionP2Controller(
        Pq=config.Pq_ATT,
        Pw=config.Pw_ATT,
        torque_limit=config.TORQUE_LIMIT
    )

    mixer = MotorMixer(
        arm_length=config.ARM_LENGTH,
        km=config.KM,
        min_thrust=config.MIN_MOTOR_THRUST,
        max_thrust=config.MAX_MOTOR_THRUST
    )

    # 5. Data Logging Arrays
    log_time = []
    log_pos_actual = []
    log_pos_des = []
    log_rpy_actual = []
    log_rpy_des = []
    log_pos_error_norm = []
    log_att_error_deg = []
    log_thrust = []
    log_torques = []
    log_motor_forces = []

    # Visual trail tracking
    prev_trail_pos = np.array(init_pos)
    trail_draw_interval = 25  # Draw trail segment every 50 ms
    step_count = 0
    total_steps = int(sim_duration / dt)

    desired_yaw_rad = np.deg2rad(config.DESIRED_YAW_DEG)
    target_rpy_rad = np.deg2rad([
        config.ATTITUDE_ROLL_DEG,
        config.ATTITUDE_PITCH_DEG,
        config.ATTITUDE_YAW_DEG
    ])

    # Interactive Camera State
    cam_dragging = False
    cam_last_x, cam_last_y = None, None

    print(f"Starting flight simulation: {total_steps} steps ({sim_duration:.1f} s)...")
    wall_start = time.perf_counter()

    for step in range(total_steps):
        t = step * dt

        # Read current rigid-body state from PyBullet
        pos_meas, orn_meas_pb = p.getBasePositionAndOrientation(drone_id)
        vel_meas_world, omega_meas_world = p.getBaseVelocity(drone_id)

        pos_meas = np.array(pos_meas, dtype=float)
        vel_meas_world = np.array(vel_meas_world, dtype=float)
        omega_meas_world = np.array(omega_meas_world, dtype=float)

        # Convert quaternion from PyBullet [x, y, z, w] to scalar-first [w, x, y, z]
        q_meas_mj = pybullet_to_mujoco_quat(orn_meas_pb)
        rpy_meas = quaternion_to_rpy(q_meas_mj)

        # Convert world angular velocity into BODY coordinates: omega_body = R^T * omega_world
        R_meas = quaternion_to_rotation_matrix(q_meas_mj)
        omega_body = R_meas.T @ omega_meas_world

        # ======================================================================
        # CONTROLLER EXECUTION ACCORDING TO EXPERIMENT MODE
        # ======================================================================
        if mode == "trajectory":
            # 1. Evaluate B-spline reference trajectory
            pos_d, vel_d, accel_d = trajectory.evaluate(t)

            # 2. Position Controller -> Commanded Thrust & Attitude Quaternion
            thrust_cmd, q_des_mj, _ = pos_controller.compute_control(
                pos=pos_meas,
                vel=vel_meas_world,
                pos_d=pos_d,
                vel_d=vel_d,
                accel_d=accel_d,
                dt=dt,
                desired_yaw_rad=desired_yaw_rad
            )

            # 3. P2 Quaternion Controller -> Body Torques
            torque_cmd, q_err = p2_controller.compute_torque(
                q_ref=q_des_mj,
                q_measured=q_meas_mj,
                omega_body=omega_body
            )
            rpy_d = quaternion_to_rpy(q_des_mj)

        else:  # attitude mode
            # Drone stays at ATTITUDE_HOLD_POSITION (hovering 1 unit above ground)
            pos_d = np.array(config.ATTITUDE_HOLD_POSITION, dtype=float)
            vel_d = np.zeros(3)

            # Target attitude from config.py (e.g. 90 deg roll, 0 deg pitch, 0 deg yaw)
            rpy_d = np.array(target_rpy_rad, dtype=float)
            q_des_mj = rpy_to_quaternion(target_rpy_rad[0], target_rpy_rad[1], target_rpy_rad[2])

            # P2 Quaternion Attitude Controller computes body torques
            torque_cmd, q_err = p2_controller.compute_torque(
                q_ref=q_des_mj,
                q_measured=q_meas_mj,
                omega_body=omega_body
            )

            # Hover thrust keeps motors running at baseline operating point
            thrust_cmd = config.HOVER_THRUST

        # 4. Motor Mixer: Allocate thrust and torques to 4 individual motors
        motor_forces, act_thrust, act_torques = mixer.mix(thrust_cmd, torque_cmd)

        # 5. Apply Forces & Torques to PyBullet Rigid Body
        for r_pos, f_i in zip(config.ROTOR_POSITIONS, motor_forces):
            p.applyExternalForce(
                objectUniqueId=drone_id,
                linkIndex=-1,
                forceObj=[0.0, 0.0, f_i],
                posObj=r_pos,
                flags=p.LINK_FRAME
            )

        # Apply aerodynamic yaw reaction torque
        yaw_reaction = config.KM * (-motor_forces[0] - motor_forces[1] + motor_forces[2] + motor_forces[3])
        p.applyExternalTorque(
            objectUniqueId=drone_id,
            linkIndex=-1,
            torqueObj=[0.0, 0.0, yaw_reaction],
            flags=p.LINK_FRAME
        )

        # 6. Step PyBullet Physics Engine
        p.stepSimulation()
        step_count += 1

        # ----------------------------------------------------------------------
        # 7. Interactive Camera Control (Mouse Click-Drag & Arrow Keys)
        # ----------------------------------------------------------------------
        # Handle mouse drag rotation
        for event in p.getMouseEvents():
            ev_type, mx, my, btn, state = event
            if ev_type == 2:  # Mouse button event
                if state & 1:  # Button pressed
                    cam_dragging = True
                    cam_last_x, cam_last_y = mx, my
                elif state & 4:  # Button released
                    cam_dragging = False
                    cam_last_x, cam_last_y = None, None
            elif ev_type == 1 and cam_dragging and cam_last_x is not None:  # Mouse move
                dx = mx - cam_last_x
                dy = my - cam_last_y
                cam_last_x, cam_last_y = mx, my
                cam_yaw = (cam_yaw + dx * 0.25) % 360.0
                cam_pitch = max(-89.0, min(89.0, cam_pitch - dy * 0.25))
                cam_info = p.getDebugVisualizerCamera()
                cam_dist = cam_info[10]
                p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_target)

        # Handle keyboard arrow keys for camera orbit
        k_events = p.getKeyboardEvents()
        cam_k_moved = False
        if 65295 in k_events and (k_events[65295] & 2):  # Left arrow
            cam_yaw = (cam_yaw - 1.2) % 360.0
            cam_k_moved = True
        if 65296 in k_events and (k_events[65296] & 2):  # Right arrow
            cam_yaw = (cam_yaw + 1.2) % 360.0
            cam_k_moved = True
        if 65297 in k_events and (k_events[65297] & 2):  # Up arrow
            cam_pitch = min(89.0, cam_pitch + 0.8)
            cam_k_moved = True
        if 65298 in k_events and (k_events[65298] & 2):  # Down arrow
            cam_pitch = max(-89.0, cam_pitch - 0.8)
            cam_k_moved = True

        if cam_k_moved:
            cam_info = p.getDebugVisualizerCamera()
            cam_dist = cam_info[10]
            p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_target)

        # Draw Dynamic Flight Path Trail (Trajectory mode)
        if mode == "trajectory" and (step_count % trail_draw_interval == 0):
            p.addUserDebugLine(
                prev_trail_pos,
                pos_meas,
                lineColorRGB=config.LINE_COLOR_ACTUAL,
                lineWidth=config.LINE_WIDTH,
                lifeTime=0
            )
            prev_trail_pos = pos_meas.copy()

        # Real-time synchronization for GUI viewer
        if config.REAL_TIME_SYNC:
            expected_wall_time = t
            elapsed = time.perf_counter() - wall_start
            if expected_wall_time > elapsed:
                time.sleep(expected_wall_time - elapsed)

        # 8. Record State Log
        pos_err = float(np.linalg.norm(pos_d - pos_meas))
        att_err = float(np.rad2deg(2.0 * np.arccos(np.clip(abs(q_err[0]), 0.0, 1.0))))

        log_time.append(t)
        log_pos_actual.append(pos_meas)
        log_pos_des.append(pos_d)
        log_rpy_actual.append(rpy_meas)
        log_rpy_des.append(rpy_d)
        log_pos_error_norm.append(pos_err)
        log_att_error_deg.append(att_err)
        log_thrust.append(act_thrust)
        log_torques.append(act_torques)
        log_motor_forces.append(motor_forces)

        # Periodic terminal update
        if step_count % int(1.0 / dt) == 0:
            print(f"  t={t:5.1f}s | Pos: [{pos_meas[0]:5.2f}, {pos_meas[1]:5.2f}, {pos_meas[2]:5.2f}] | "
                  f"PosErr: {pos_err:5.2f}m | RPY: [{np.rad2deg(rpy_meas[0]):+5.1f}°, {np.rad2deg(rpy_meas[1]):+5.1f}°, {np.rad2deg(rpy_meas[2]):+5.1f}°] | "
                  f"Thrust: {act_thrust:5.2f}N")

    # Simulation complete
    print("\nSimulation completed successfully!")
    p.disconnect()

    # Convert logged lists to numpy arrays
    time_arr = np.array(log_time)
    pos_act_arr = np.array(log_pos_actual)
    pos_des_arr = np.array(log_pos_des)
    rpy_act_arr = np.rad2deg(np.array(log_rpy_actual))
    rpy_des_arr = np.rad2deg(np.array(log_rpy_des))
    pos_err_arr = np.array(log_pos_error_norm)
    att_err_arr = np.array(log_att_error_deg)
    thrust_arr = np.array(log_thrust)
    torque_arr = np.array(log_torques)
    motors_arr = np.array(log_motor_forces)

    # 9. Compute Performance Metrics
    rmse_x = np.sqrt(np.mean((pos_act_arr[:, 0] - pos_des_arr[:, 0])**2))
    rmse_y = np.sqrt(np.mean((pos_act_arr[:, 1] - pos_des_arr[:, 1])**2))
    rmse_z = np.sqrt(np.mean((pos_act_arr[:, 2] - pos_des_arr[:, 2])**2))
    rmse_pos = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)
    mean_att_err = np.mean(att_err_arr)
    max_drift_xy = np.max(np.hypot(pos_act_arr[:, 0] - pos_des_arr[0, 0], pos_act_arr[:, 1] - pos_des_arr[0, 1]))

    print("\n" + "=" * 65)
    print(" FLIGHT PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  3D Position RMSE          : {rmse_pos:.4f} m")
    print(f"    - X Axis RMSE           : {rmse_x:.4f} m")
    print(f"    - Y Axis RMSE           : {rmse_y:.4f} m")
    print(f"    - Z Axis RMSE           : {rmse_z:.4f} m")
    if mode == "attitude":
        print(f"  Max Horizontal Drift (XY) : {max_drift_xy:.4f} m (Locked at {config.ATTITUDE_HOLD_POSITION})")
        print(f"  Final Position            : [{pos_act_arr[-1,0]:.2f}, {pos_act_arr[-1,1]:.2f}, {pos_act_arr[-1,2]:.2f}] m")
        print(f"  Final Actual RPY          : [{rpy_act_arr[-1,0]:+.1f}°, {rpy_act_arr[-1,1]:+.1f}°, {rpy_act_arr[-1,2]:+.1f}°]")
        print(f"  Final Commanded RPY       : [{rpy_des_arr[-1,0]:+.1f}°, {rpy_des_arr[-1,1]:+.1f}°, {rpy_des_arr[-1,2]:+.1f}°]")
        print(f"  Mean Attitude Error       : {mean_att_err:.2f}°")
    print(f"  Mean Thrust Output        : {np.mean(thrust_arr):.2f} N (Hover ~ {config.HOVER_THRUST:.2f} N)")
    print(f"  Max Motor Force           : {np.max(motors_arr):.2f} N (Limit = {config.MAX_MOTOR_THRUST:.2f} N)")
    print("=" * 65)

    # 10. Save Logs and Generate Plots
    save_results_and_plots(
        time_arr, pos_act_arr, pos_des_arr,
        rpy_act_arr, rpy_des_arr,
        pos_err_arr, att_err_arr,
        thrust_arr, torque_arr, motors_arr,
        waypoints, mode
    )


# ==============================================================================
# LOG SAVING & PLOT GENERATION
# ==============================================================================

def save_results_and_plots(time_arr, pos_act, pos_des, rpy_act, rpy_des,
                           pos_err, att_err,
                           thrust, torques, motors, waypoints, mode):
    """
    Saves CSV telemetry log and renders high-quality tracking plots in results/.
    For B-spline trajectory mode, opens an interactive 3D Matplotlib window
    so the user can click and drag to rotate the 3D plot.
    """
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Save telemetry CSV
    csv_path = os.path.join(results_dir, "flight_log.csv")
    csv_header = (
        "time,x_meas,y_meas,z_meas,x_des,y_des,z_des,"
        "roll_meas_deg,pitch_meas_deg,yaw_meas_deg,"
        "roll_des_deg,pitch_des_deg,yaw_des_deg,"
        "pos_error_norm,att_error_deg,"
        "thrust_N,tau_x,tau_y,tau_z,f1,f2,f3,f4"
    )
    csv_data = np.column_stack([
        time_arr,
        pos_act, pos_des,
        rpy_act, rpy_des,
        pos_err, att_err,
        thrust, torques, motors
    ])
    np.savetxt(csv_path, csv_data, delimiter=",", header=csv_header, comments="")
    print(f"\n[Saved] Telemetry log saved to: {csv_path}")

    # Plot styling
    plt.rcParams.update({
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'figure.autolayout': True
    })

    # --------------------------------------------------------------------------
    # Plot 1: Position Tracking / Position Hold
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    coords = ['X', 'Y', 'Z']
    colors_act = ['#1f77b4', '#2ca02c', '#9467bd']

    for i in range(3):
        ax = axes[i]
        ax.plot(time_arr, pos_des[:, i], 'r--', linewidth=2.0, label=f'Desired {coords[i]}')
        ax.plot(time_arr, pos_act[:, i], color=colors_act[i], linewidth=2.0, label=f'Actual {coords[i]}')
        ax.set_ylabel(f'{coords[i]} Position [m]')
        ax.legend(loc='upper right')
        rmse_axis = np.sqrt(np.mean((pos_act[:, i] - pos_des[:, i])**2))
        ax.set_title(f'{coords[i]}-Axis Tracking (RMSE = {rmse_axis:.3f} m)', fontsize=11, fontweight='bold')

    axes[2].set_xlabel('Time [s]')
    title_str = 'Attitude Mode Position Hold (Hover 1 unit above ground)' if mode == 'attitude' else 'Trajectory Position Tracking'
    fig.suptitle(title_str, fontsize=14, fontweight='bold')

    if mode == "attitude":
        plot_pos_path = os.path.join(results_dir, "position_hold.png")
    else:
        plot_pos_path = os.path.join(results_dir, "tracking_positions.png")

    fig.savefig(plot_pos_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] Position plot saved to: {plot_pos_path}")

    if mode == "attitude":
        fig_copy_path = os.path.join(results_dir, "tracking_positions.png")
        import shutil
        shutil.copyfile(plot_pos_path, fig_copy_path)

    # --------------------------------------------------------------------------
    # Plot 2: Attitude Tracking (Roll, Pitch, Yaw vs Time)
    # --------------------------------------------------------------------------
    fig_att, axes_att = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    rpy_names = ['Roll', 'Pitch', 'Yaw']
    rpy_colors = ['#d62728', '#2ca02c', '#1f77b4']

    for i in range(3):
        ax = axes_att[i]
        ax.plot(time_arr, rpy_des[:, i], 'r--', linewidth=2.0, label=f'Desired {rpy_names[i]}')
        ax.plot(time_arr, rpy_act[:, i], color=rpy_colors[i], linewidth=2.0, label=f'Actual {rpy_names[i]}')
        ax.set_ylabel(f'{rpy_names[i]} [deg]')
        ax.legend(loc='upper right')
        ax.set_title(f'{rpy_names[i]} Angle Tracking', fontsize=11, fontweight='bold')

    axes_att[2].set_xlabel('Time [s]')
    fig_att.suptitle('Attitude Tracking (Quaternion P2 Controller)', fontsize=14, fontweight='bold')
    plot_att_path = os.path.join(results_dir, "attitude_tracking.png")
    fig_att.savefig(plot_att_path, dpi=300)
    plt.close(fig_att)
    print(f"[Saved] Attitude tracking plot saved to: {plot_att_path}")

    # --------------------------------------------------------------------------
    # Plot 3: Control Inputs & Motor Forces
    # --------------------------------------------------------------------------
    fig_eff, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Total thrust
    ax1.plot(time_arr, thrust, color='#e67e22', linewidth=2.0, label='Total Commanded Thrust')
    ax1.axhline(config.HOVER_THRUST, color='black', linestyle=':', label=f'Hover Thrust ({config.HOVER_THRUST:.2f} N)')
    ax1.set_ylabel('Thrust [N]')
    ax1.legend(loc='upper right')
    ax1.set_title('Collective Thrust Output', fontsize=11, fontweight='bold')

    # Individual Motor Forces
    motor_labels = ['M1 (Front)', 'M2 (Rear)', 'M3 (Left)', 'M4 (Right)']
    motor_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for m in range(4):
        ax2.plot(time_arr, motors[:, m], color=motor_colors[m], linewidth=1.5, label=motor_labels[m])
    ax2.axhline(config.MAX_MOTOR_THRUST, color='red', linestyle='--', alpha=0.7, label=f'Motor Max ({config.MAX_MOTOR_THRUST} N)')
    ax2.axhline(config.MIN_MOTOR_THRUST, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Motor Force [N]')
    ax2.legend(loc='upper right', ncol=2)
    ax2.set_title('Individual Rotor Forces (Motor Mixer)', fontsize=11, fontweight='bold')

    fig_eff.suptitle('Control Efforts and Actuator Allocations', fontsize=14, fontweight='bold')
    plot_effort_path = os.path.join(results_dir, "control_efforts.png")
    fig_eff.savefig(plot_effort_path, dpi=300)
    plt.close(fig_eff)
    print(f"[Saved] Control efforts plot saved to: {plot_effort_path}")

    # --------------------------------------------------------------------------
    # Plot 4: 3D Trajectory (Trajectory Mode only) - Opens Interactive Window
    # --------------------------------------------------------------------------
    if mode == "trajectory":
        fig_3d = plt.figure(figsize=(10, 8))
        ax = fig_3d.add_subplot(111, projection='3d')

        ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'r--', linewidth=2.5, label='Desired B-Spline Path')
        ax.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'b-', linewidth=2.0, label='Actual Drone Flight Path')

        if waypoints is not None and len(waypoints) > 0:
            wp_arr = np.array(waypoints)
            ax.scatter(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2], c='#00a8ff', s=70, depthshade=False, label='Waypoints')
            ax.scatter([wp_arr[0, 0]], [wp_arr[0, 1]], [wp_arr[0, 2]], c='green', s=130, marker='o', label='Start (WP1)')
            ax.scatter([wp_arr[-1, 0]], [wp_arr[-1, 1]], [wp_arr[-1, 2]], c='red', s=130, marker='^', label='Goal (Final WP)')
            for k, pt in enumerate(wp_arr):
                ax.text(pt[0], pt[1], pt[2] + 0.15, f'WP{k+1}', fontsize=10, fontweight='bold')

        ax.plot(pos_act[:, 0], pos_act[:, 1], np.zeros_like(pos_act[:, 2]), color='gray', linestyle=':', alpha=0.5, label='Ground Projection')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Altitude Z [m]')
        ax.set_title('3D Quadrotor Trajectory Following Comparison\n(Click & drag mouse to rotate 3D view)', fontsize=13, fontweight='bold')
        ax.legend(loc='best')

        plot_3d_path = os.path.join(results_dir, "trajectory_3d.png")
        fig_3d.savefig(plot_3d_path, dpi=300)
        print(f"[Saved] 3D trajectory plot saved to: {plot_3d_path}")

        # Open interactive matplotlib window so user can rotate in 3D
        print("\n" + "=" * 65)
        print(" [Interactive Plot] Opening 3D Trajectory Window...")
        print(" Click and drag with your mouse to rotate the 3D plot.")
        print(" Close the plot window when done.")
        print("=" * 65)
        plt.show()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyBullet Quadrotor Simulation with P2 Attitude Controller")
    parser.add_argument("--mode", type=str, choices=["trajectory", "attitude"], default=None,
                        help="Select experiment mode: 'trajectory' (default) or 'attitude'")
    args = parser.parse_args()

    run_simulation(flight_mode=args.mode)
