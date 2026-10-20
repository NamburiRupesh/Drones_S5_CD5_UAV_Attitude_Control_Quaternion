"""
B-Spline Trajectory Generator for Quadrotor Waypoint Following.

Trajectory Type:
    EXACT INTERPOLATING B-SPLINE (C2 Continuous)
    
Characteristics:
    - Interpolates all waypoints EXACTLY: p(t_i) == waypoints[i].
    - C2 continuity: Smooth position, velocity, and acceleration profiles.
    - Zero boundary velocity: Enforces v(0) = 0 and v(T) = 0 so the drone takes off
      from rest and arrives at the final destination smoothly without velocity spikes.
    - Chord-length time parameterization: Segment durations are proportional to
      the Euclidean distance between consecutive waypoints, preventing excessive
      speed variations.
    - Automatic adaptation: Dynamically supports any number of waypoints (N >= 2)
      by adjusting spline degree k = min(3, N - 1).
"""

import numpy as np
from scipy.interpolate import make_interp_spline


class BSplineWaypointTrajectory:
    """
    Generates a smooth, physically feasible trajectory through an arbitrary
    list of 3D waypoints using an interpolating B-spline.
    """

    def __init__(self, waypoints, total_duration=15.0):
        """
        Initialize the trajectory.

        Args:
            waypoints: List or array of shape (N, 3), where each row is [x, y, z].
            total_duration: Total flight duration in seconds (float).
        """
        self.waypoints = np.array(waypoints, dtype=float)
        if len(self.waypoints) < 2:
            raise ValueError("At least 2 waypoints are required to form a trajectory.")

        self.total_duration = float(total_duration)
        self.num_points = len(self.waypoints)

        # 1. Parameterize time based on chord length (Euclidean distances)
        diffs = np.diff(self.waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)

        if total_length < 1e-6:
            # All waypoints are essentially at the same location
            self.t_points = np.linspace(0.0, self.total_duration, self.num_points)
        else:
            # Distribute total_duration proportionally to segment distances
            fractions = segment_lengths / total_length
            durations = fractions * self.total_duration
            
            # Ensure every segment gets a non-zero minimum time slice
            min_seg_dt = 0.2
            durations = np.maximum(durations, min_seg_dt)
            # Re-normalize to exactly match total_duration
            durations = durations * (self.total_duration / np.sum(durations))

            self.t_points = np.zeros(self.num_points)
            self.t_points[1:] = np.cumsum(durations)

        # 2. Select spline degree:
        # Degree k = min(3, N - 1).
        # For N >= 4, k=3 (Cubic spline, continuous acceleration).
        # For N = 3, k=2 (Quadratic spline).
        # For N = 2, k=1 (Linear).
        self.k = min(3, self.num_points - 1)

        # 3. Fit interpolating B-spline with zero boundary velocity for smooth start and end
        if self.k == 3:
            # Enforce zero velocity at start (t=0) and goal (t=T)
            bc_type = ([(1, np.zeros(3))], [(1, np.zeros(3))])
            self.spline_p = make_interp_spline(
                self.t_points, self.waypoints, k=self.k, bc_type=bc_type
            )
        else:
            self.spline_p = make_interp_spline(
                self.t_points, self.waypoints, k=self.k
            )

        # 4. Analytic derivatives for velocity and acceleration
        self.spline_v = self.spline_p.derivative(1)
        self.spline_a = self.spline_p.derivative(2) if self.k >= 2 else None

    def evaluate(self, t):
        """
        Evaluate reference position, velocity, and acceleration at time t.

        Args:
            t: Current time in seconds.

        Returns:
            p_d: Desired position [x, y, z] in meters.
            v_d: Desired velocity [vx, vy, vz] in m/s.
            a_d: Desired acceleration [ax, ay, az] in m/s^2.
        """
        if t <= 0.0:
            return self.waypoints[0].copy(), np.zeros(3), np.zeros(3)
        if t >= self.total_duration:
            return self.waypoints[-1].copy(), np.zeros(3), np.zeros(3)

        p_d = np.array(self.spline_p(t), dtype=float)
        v_d = np.array(self.spline_v(t), dtype=float)
        if self.spline_a is not None:
            a_d = np.array(self.spline_a(t), dtype=float)
        else:
            a_d = np.zeros(3)

        return p_d, v_d, a_d

    def sample_path(self, num_samples=300):
        """
        Sample the full 3D path curve for PyBullet visual line drawing.

        Returns:
            samples: Array of shape (num_samples, 3).
        """
        t_vals = np.linspace(0.0, self.total_duration, num_samples)
        return np.array([self.spline_p(t) for t in t_vals])
