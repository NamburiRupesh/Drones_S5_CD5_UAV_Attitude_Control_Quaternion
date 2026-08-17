import numpy as np


class QuaternionP2Controller:

    def __init__(self, Pq=20.0, Pw=4.0, torque_limit=4.0):
        self.Pq = Pq
        self.Pw = Pw
        self.torque_limit = torque_limit

    @staticmethod
    def normalize(q):
        q = np.asarray(q, dtype=float)
        return q / np.linalg.norm(q)

    @staticmethod
    def conjugate(q):
        # q = [q0, q1, q2, q3]
        return np.array([
            q[0],
            -q[1],
            -q[2],
            -q[3]
        ])

    @staticmethod
    def multiply(q1, q2):
        """
        Quaternion multiplication.

        Quaternion format:
        [q0, q1, q2, q3]
        where q0 is scalar.
        """

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,

            w1*x2 + x1*w2 + y1*z2 - z1*y2,

            w1*y2 - x1*z2 + y1*w2 + z1*x2,

            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def compute_error(self, q_ref, q_measured, shortest_path=True):

        q_ref = self.normalize(q_ref)
        q_measured = self.normalize(q_measured)

        # Quaternion error:
        # q_err = q_ref ⊗ q_measured*
        q_err = self.multiply(
            q_ref,
            self.conjugate(q_measured)
        )

        # For normal attitude tracking, use the shortest rotation.
        #
        # IMPORTANT:
        # This must be disabled for the 360-degree flip because
        # the flip intentionally follows the long/continuous path.
        if shortest_path and q_err[0] < 0:
            q_err = -q_err

        # Vector part of quaternion error
        axis_error = q_err[1:4]

        return q_err, axis_error

    def compute_torque(
        self,
        q_ref,
        q_measured,
        omega,
        shortest_path=True
    ):

        q_err, axis_error = self.compute_error(
            q_ref,
            q_measured,
            shortest_path=shortest_path
        )

        omega = np.asarray(omega, dtype=float)

        # Quaternion P² control law
        torque = (
            -self.Pq * axis_error
            -self.Pw * omega
        )

        # Torque saturation
        torque = np.clip(
            torque,
            -self.torque_limit,
            self.torque_limit
        )

        return torque, q_err