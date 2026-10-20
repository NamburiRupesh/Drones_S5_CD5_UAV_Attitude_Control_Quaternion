"""
Quadrotor Live Telemetry Plotter
---------------------------------
High-performance dynamic telemetry dashboard for real-time quadrotor flight visualization.

Features:
- Configurable plot selection: choose any combination of channels:
    ["x", "y", "z", "error", "roll", "pitch", "yaw", "thrust"]
- Automatic window snapping:
    * Left side (~62% screen width): MuJoCo 3D Physics Simulation
    * Right side (~38% screen width): Dynamic Telemetry Dashboard
- Dedicated subprocess pipeline via subprocess.Popen for guaranteed zero lag.
- Vertical N-subplot layout dynamically adapting to the selected channels.
"""

import os
import sys
import subprocess
import threading
import queue
import numpy as np


def snap_mujoco_window(sim_fraction=0.62):
    """Snaps the active MuJoCo viewer window to the left side of the display."""
    try:
        import ctypes
        import ctypes.wintypes
        import time

        user32 = ctypes.windll.user32
        rect = ctypes.wintypes.RECT()
        user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
        sw = rect.right - rect.left
        sh = rect.bottom - rect.top
        w_sim = int(sw * sim_fraction)

        def find_and_snap():
            found = False
            def enum_cb(hwnd, _):
                nonlocal found
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        if "MuJoCo" in buff.value:
                            user32.SetWindowPos(hwnd, 0, rect.left, rect.top, w_sim, sh, 0x0040)
                            found = True
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
            return found

        for _ in range(6):
            if find_and_snap():
                break
            time.sleep(0.08)
    except Exception as e:
        print(f"[LivePlotter] Note on MuJoCo window snapping: {e}")


def run_plotter_gui(channels=None):
    """Runs the dynamic telemetry dashboard for the chosen channels."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not channels:
        channels = ["x", "y", "z", "error"]

    channels = [c.lower().strip() for c in channels]
    valid_channels = ["x", "y", "z", "error", "roll", "pitch", "yaw", "thrust"]
    channels = [c for c in channels if c in valid_channels]
    if not channels:
        channels = ["x", "y", "z", "error"]

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    num_plots = len(channels)
    fig, axes = plt.subplots(num_plots, 1, figsize=(7, max(4, num_plots * 2.5)), sharex=False)
    if num_plots == 1:
        axes = [axes]

    channel_names = ", ".join(c.upper() for c in channels)
    fig.canvas.manager.set_window_title(f"Telemetry Dashboard [{channel_names}]")

    # Snap Plotter Window to the Right Side (occupying 38% screen width)
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        rect = ctypes.wintypes.RECT()
        user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
        sw = rect.right - rect.left
        sh = rect.bottom - rect.top
        sim_fraction = 0.62
        w_sim = int(sw * sim_fraction)
        w_plot = sw - w_sim

        manager = plt.get_current_fig_manager()
        manager.window.geometry(f"{w_plot}x{sh}+{rect.left + w_sim}+{rect.top}")
        manager.window.attributes("-topmost", True)
        manager.window.after(600, lambda: manager.window.attributes("-topmost", False))
    except Exception:
        pass

    # Telemetry storage
    times = []
    x_act, x_des = [], []
    y_act, y_des = [], []
    z_act, z_des = [], []
    errors = []
    rolls, pitches, yaws = [], [], []
    thrusts = []

    # Configure each requested channel subplot
    plot_elements = {}
    for ax, ch in zip(axes, channels):
        ax.grid(True, linestyle=":", alpha=0.6)

        if ch == "x":
            (l_act,) = ax.plot([], [], color="#1f77b4", linewidth=1.5, label="X Actual")
            (l_des,) = ax.plot([], [], color="#ff7f0e", linestyle="--", linewidth=1.5, label="X Desired")
            ax.set_title("X Position Tracking", fontsize=10, fontweight="bold", pad=2)
            ax.set_ylabel("X [m]", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            plot_elements["x"] = (l_act, l_des, ax)

        elif ch == "y":
            (l_act,) = ax.plot([], [], color="#1f77b4", linewidth=1.5, label="Y Actual")
            (l_des,) = ax.plot([], [], color="#ff7f0e", linestyle="--", linewidth=1.5, label="Y Desired")
            ax.set_title("Y Position Tracking", fontsize=10, fontweight="bold", pad=2)
            ax.set_ylabel("Y [m]", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            plot_elements["y"] = (l_act, l_des, ax)

        elif ch == "z":
            (l_act,) = ax.plot([], [], color="#1f77b4", linewidth=1.5, label="Z Actual")
            (l_des,) = ax.plot([], [], color="#ff7f0e", linestyle="--", linewidth=1.5, label="Z Desired")
            ax.axhline(0.0, color="#d62728", linestyle=":", linewidth=1.3, label="Ground (0m)")
            ax.set_title("Z Altitude Tracking", fontsize=10, fontweight="bold", pad=2)
            ax.set_ylabel("Z [m]", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            plot_elements["z"] = (l_act, l_des, ax)

        elif ch == "error":
            (l_err,) = ax.plot([], [], color="#d62728", linewidth=1.5, label="||p - p_des||")
            ax.set_title("3D Tracking Error", fontsize=10, fontweight="bold", pad=2)
            ax.set_ylabel("Error [m]", fontsize=9)
            err_box = ax.text(
                0.96, 0.88, "Err: 0.000 m",
                transform=ax.transAxes,
                fontsize=8, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc")
            )
            ax.legend(loc="upper left", fontsize=8)
            plot_elements["error"] = (l_err, err_box, ax)

        elif ch == "roll":
            (l_roll,) = ax.plot([], [], color="#2ca02c", linewidth=1.5, label="Roll")
            ax.set_title("Roll Attitude", fontsize=10, fontweight="bold", pad=2)
            ax.set_ylabel("Roll [deg]", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            plot_elements["roll"] = (l_roll, ax)

        elif ch == "pitch":
            (l_pitch,) = ax.plot([], [], color="#2ca02c", linewidth=1.5, label="Pitch")
            ax.set_title("Pitch Attitude", fontsize=10, fontweight="bold", pad=2)
            ax.set_ylabel("Pitch [deg]", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            plot_elements["pitch"] = (l_pitch, ax)

        elif ch == "yaw":
            (l_yaw,) = ax.plot([], [], color="#2ca02c", linewidth=1.5, label="Yaw")
            ax.set_title("Yaw Attitude", fontsize=10, fontweight="bold", pad=2)
            ax.set_ylabel("Yaw [deg]", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            plot_elements["yaw"] = (l_yaw, ax)

        elif ch == "thrust":
            (l_thrust,) = ax.plot([], [], color="#9467bd", linewidth=1.5, label="Thrust")
            ax.axhline(9.81, color="gray", linestyle="--", linewidth=1.2, label="Hover (~9.8 N)")
            ax.set_title("Thrust Command", fontsize=10, fontweight="bold", pad=2)
            ax.set_ylabel("Thrust [N]", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            plot_elements["thrust"] = (l_thrust, ax)

    axes[-1].set_xlabel("Time [s]", fontsize=9)
    fig.suptitle("Real-Time Telemetry Dashboard", fontsize=11, fontweight="bold")
    plt.tight_layout()

    data_queue = queue.Queue()
    is_complete = False

    def reader_thread():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            data_queue.put(line)
            if line == "DONE":
                break

    t_reader = threading.Thread(target=reader_thread, daemon=True)
    t_reader.start()

    def update(frame):
        nonlocal is_complete
        updated = False

        while not data_queue.empty():
            try:
                line = data_queue.get_nowait()
            except queue.Empty:
                break

            if line == "DONE":
                is_complete = True
                fig.canvas.manager.set_window_title(f"Telemetry Dashboard [{channel_names}] [COMPLETED]")
                continue

            try:
                parts = [float(v) for v in line.split(",")]
                if len(parts) >= 7:
                    t, px, py, pz, dx, dy, dz = parts[:7]
                    r = parts[7] if len(parts) > 7 else 0.0
                    p = parts[8] if len(parts) > 8 else 0.0
                    y = parts[9] if len(parts) > 9 else 0.0
                    th = parts[10] if len(parts) > 10 else 0.0

                    w_flag = bool(int(parts[11])) if len(parts) > 11 else False
                    w_mag = float(parts[12]) if len(parts) > 12 else 3.0

                    err = np.sqrt((px - dx)**2 + (py - dy)**2 + (pz - dz)**2)
                    times.append(t)
                    x_act.append(px); x_des.append(dx)
                    y_act.append(py); y_des.append(dy)
                    z_act.append(pz); z_des.append(dz)
                    errors.append(err)
                    rolls.append(r); pitches.append(p); yaws.append(y)
                    thrusts.append(th)
                    updated = True
            except ValueError:
                continue

        if updated and len(times) > 0:
            if "x" in plot_elements:
                l_act, l_des, ax = plot_elements["x"]
                l_act.set_data(times, x_act)
                l_des.set_data(times, x_des)
                ax.relim(); ax.autoscale_view()

            if "y" in plot_elements:
                l_act, l_des, ax = plot_elements["y"]
                l_act.set_data(times, y_act)
                l_des.set_data(times, y_des)
                ax.relim(); ax.autoscale_view()

            if "z" in plot_elements:
                l_act, l_des, ax = plot_elements["z"]
                l_act.set_data(times, z_act)
                l_des.set_data(times, z_des)
                ax.relim(); ax.autoscale_view()

            if "error" in plot_elements:
                l_err, err_box, ax = plot_elements["error"]
                l_err.set_data(times, errors)
                cur_err = errors[-1]
                rms_err = np.sqrt(np.mean(np.square(errors)))
                max_err = np.max(errors)
                wind_label = f"\n[WIND: {w_mag:.1f}N]" if w_flag else ""
                err_box.set_text(f"Cur: {cur_err:.3f} m\nRMS: {rms_err:.3f} m{wind_label}")
                if w_flag:
                    err_box.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", alpha=0.9, edgecolor="#ffc107"))
                else:
                    err_box.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"))
                ax.relim(); ax.autoscale_view()

            if "roll" in plot_elements:
                l_roll, ax = plot_elements["roll"]
                l_roll.set_data(times, rolls)
                ax.relim(); ax.autoscale_view()

            if "pitch" in plot_elements:
                l_pitch, ax = plot_elements["pitch"]
                l_pitch.set_data(times, pitches)
                ax.relim(); ax.autoscale_view()

            if "yaw" in plot_elements:
                l_yaw, ax = plot_elements["yaw"]
                l_yaw.set_data(times, yaws)
                ax.relim(); ax.autoscale_view()

            if "thrust" in plot_elements:
                l_thrust, ax = plot_elements["thrust"]
                l_thrust.set_data(times, thrusts)
                ax.relim(); ax.autoscale_view()

        return axes

    ani = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()


class LiveTelemetryPlotter:
    """
    Spawns and manages the independent live telemetry plotting process via standard subprocess.
    Accepts arbitrary channels to plot (e.g. ['z', 'error'] or ['x', 'y', 'z', 'error', 'thrust']).
    """

    def __init__(self, enabled=True, channels=None, sim_fraction=0.62):
        self.enabled = enabled
        self.channels = [str(c).lower().strip() for c in (channels or ["x", "y", "z", "error"])]
        self.sim_fraction = sim_fraction
        self.process = None

    def start(self):
        """Starts the live plotter window with chosen channels and snaps both windows side-by-side."""
        if not self.enabled:
            return

        # 1. Snap MuJoCo window to the left side
        snap_mujoco_window(sim_fraction=self.sim_fraction)

        # 2. Launch plotter subprocess snapped to the right side
        script_path = os.path.abspath(__file__)
        cmd = [sys.executable, script_path] + self.channels
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=None,
                text=True,
                bufsize=1
            )
            ch_str = ", ".join(self.channels)
            print(f"[LivePlotter] Telemetry Dashboard launched for [{ch_str}] (PID: {self.process.pid})")
        except Exception as e:
            print(f"[LivePlotter] Failed to spawn live plotter: {e}")
            self.process = None

    def send(self, t, pos, pos_des, rpy=None, thrust=0.0, wind_active=False, wind_mag=0.0):
        """
        Sends an instantaneous telemetry snapshot to the plotter process.
        Non-blocking to ensure zero impact on physics.
        """
        if not self.enabled or self.process is None or self.process.poll() is not None:
            return

        try:
            r = np.rad2deg(rpy[0]) if rpy is not None else 0.0
            p = np.rad2deg(rpy[1]) if rpy is not None else 0.0
            y = np.rad2deg(rpy[2]) if rpy is not None else 0.0
            w_flag = 1 if wind_active else 0
            line = (
                f"{t:.4f},{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f},"
                f"{pos_des[0]:.4f},{pos_des[1]:.4f},{pos_des[2]:.4f},"
                f"{r:.2f},{p:.2f},{y:.2f},{thrust:.2f},{w_flag},{wind_mag:.2f}\n"
            )
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except Exception:
            pass

    def stop(self):
        """Signals completion of the simulation to the plotter window."""
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write("DONE\n")
                self.process.stdin.flush()
            except Exception:
                pass


if __name__ == "__main__":
    cli_channels = sys.argv[1:] if len(sys.argv) > 1 else ["x", "y", "z", "error"]
    run_plotter_gui(cli_channels)
