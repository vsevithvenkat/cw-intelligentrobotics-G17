# scara_fk.py
import math
import numpy as np

# Official coursework joint limits (used in Task 2/3 [-1, 1] normalisation)
# Elbow-up restriction applied: arm_motor in [0.0, 0.83]
SCARA_JOINT_LIMITS = np.array([
    [-0.73, 0.73],  # q1: base_arm_motor (rad)
    [ 0.00, 0.83],  # q2: arm_motor (rad, positive only for elbow-up)
    [-0.20, 0.00],  # q4: shaft_linear_motor (m)
], dtype=float)

# Option 1 Assumption: Collision-free operational limit for q4.
# In Webots R2023b, commanding q4 < -0.1729 m causes the vertical shaft tip
# to physically contact the floor/stop at z_TCP = 0.0093 m. Restricting
# sampled poses to q4 >= -0.170 m ensures all sampled configurations are
# physically reachable without floor collision.
SCARA_SAFE_Q4_MIN = -0.170


def Rz(theta):
    """Rotation matrix around the Z-axis."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)


def scara_forward_kinematics(joints):
    """
    Calculates the Forward Kinematics (x, y, z) position of the Epson SCARA TCP
    in the top-level ScaraT6 frame.

    Parameters:
        joints (list or np.ndarray): Active joint values [q1, q2, q4]
            q1: base_arm_motor (radians, [-0.73, 0.73])
            q2: arm_motor (radians, [-0.83, 0.83], > 0 for elbow-up)
            q4: shaft_linear_motor (metres, [-0.20, 0.0])
    Returns:
        np.ndarray: 3D TCP position [x, y, z] in metres.
    """
    if len(joints) != 3:
        raise ValueError(
            "Input must contain exactly 3 active joint values: "
            "[base_arm_motor, arm_motor, shaft_linear_motor]."
        )

    q1, q2, q4 = float(joints[0]), float(joints[1]), float(joints[2])

    p_base_to_j1 = np.array([0.060, 0.0, 0.220], dtype=float)
    v_j1_to_j2 = np.array([0.310, 0.005, -0.004], dtype=float)
    v_j2_to_tcp = np.array([0.2545, -0.005, q4 - 0.034], dtype=float)

    p_tcp = p_base_to_j1 + Rz(q1) @ (v_j1_to_j2 + Rz(q2) @ v_j2_to_tcp)
    return p_tcp


def sample_scara_collision_free(n_samples, rng):
    """
    Generates n_samples collision-free elbow-up configurations for SCARA
    within q1 in [-0.73, 0.73], q2 in [0.0, 0.83], q4 in [-0.170, 0.0].
    """
    return np.column_stack([
        rng.uniform(-0.73, 0.73, n_samples),
        rng.uniform(0.0, 0.83, n_samples),
        rng.uniform(SCARA_SAFE_Q4_MIN, 0.0, n_samples)
    ])


if __name__ == "__main__":
    test_joints = [0.0, 0.4, -0.10]
    tcp_pos = scara_forward_kinematics(test_joints)
    print(f"Active joints [q1, q2, q4]: {test_joints}")
    print(f"SCARA FK TCP (m): x={tcp_pos[0]:.6f}, y={tcp_pos[1]:.6f}, z={tcp_pos[2]:.6f}")