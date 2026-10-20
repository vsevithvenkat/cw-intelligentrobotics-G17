# ned_fk.py
import math
import numpy as np

# Link offsets from the Coursework Specification (in metres)
H_SHOULDER = 0.1065 + 0.0650  # 0.1715 m (Base frame to shoulder height)
A = 0.2210                    # Shoulder to elbow vertical component
B = 0.0120                    # Shoulder to elbow horizontal component
U = 0.03185                   # Elbow to teaching TCP vertical component
V = 0.23000                   # Elbow to teaching TCP horizontal component

# Official coursework joint limits (for Task 2/3 [-1, 1] normalisation)
# Elbow-up restriction applied: joint_3 in [0.0, 1.3]
NED_JOINT_LIMITS = np.array([
    [-2.8, 2.8],   # q1: joint_1
    [-0.8, 0.8],   # q2: joint_2
    [ 0.0, 1.3],   # q3: joint_3 (positive only for elbow-up)
], dtype=float)

# Collision-free operational sampling limits (Option 1 Assumption):
# Restricts extreme combined shoulder+elbow downward pitch that causes the
# wrist/gripper assembly (which extends beyond the TCP) to contact the floor/base.
NED_MIN_SAFE_TCP_Z = 0.120  # metres above the floor plane (z = 0)
NED_MAX_PITCH_SUM = 1.15    # rad: prevents wrist/hand self-collision & floor contact


def Rx(theta):
    """Rotation matrix around the X-axis."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c]
    ], dtype=float)


def Rz(theta):
    """Rotation matrix around the Z-axis."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [  c,  -s, 0.0],
        [  s,   c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)


def ned_forward_kinematics(joints):
    """
    Calculates the Forward Kinematics (x, y, z) position of the Niryo Ned TCP
    in the ARM base frame.

    Parameters:
        joints (list or np.ndarray): Active joint values [joint_1, joint_2, joint_3] in radians.
    Returns:
        np.ndarray: [x, y, z] TCP coordinates in metres.
    """
    if len(joints) != 3:
        raise ValueError("Input must contain exactly 3 active joint values: [joint_1, joint_2, joint_3].")

    q1, q2, q3 = float(joints[0]), float(joints[1]), float(joints[2])

    # Base to shoulder offset vector
    p_base_to_shoulder = np.array([0.0, 0.0, H_SHOULDER])

    # At q1 = 0, the Ned arm points along the positive Y-axis.
    # Therefore, at q2 = q3 = 0, the horizontal components (B, V) lie along +Y
    # and the vertical components (A, U) lie along +Z.
    v_shoulder_to_elbow = np.array([0.0, B, A])
    v_elbow_to_tcp = np.array([0.0, V, U])

    # Pitch rotations around X-axis, followed by waist rotation around Z-axis
    p_tcp = p_base_to_shoulder + Rz(q1) @ (
        Rx(-q2) @ (v_shoulder_to_elbow + Rx(-q3) @ v_elbow_to_tcp)
    )

    return p_tcp


def is_ned_pose_collision_free(joints):
    """
    Checks whether a candidate [q1, q2, q3] configuration lies within the
    collision-free physical workspace of the Niryo Ned in Webots.
    """
    q1, q2, q3 = float(joints[0]), float(joints[1]), float(joints[2])
    if not (-2.8 <= q1 <= 2.8 and -0.8 <= q2 <= 0.8 and 0.0 <= q3 <= 1.3):
        return False
    if (q2 + q3) > NED_MAX_PITCH_SUM:
        return False
    tcp = ned_forward_kinematics(joints)
    return bool(tcp[2] >= NED_MIN_SAFE_TCP_Z)


def sample_ned_collision_free(n_samples, rng):
    """Generates n_samples collision-free elbow-up configurations for Ned."""
    valid_joints = []
    while len(valid_joints) < n_samples:
        q = [
            rng.uniform(-2.8, 2.8),
            rng.uniform(-0.8, 0.7),
            rng.uniform(0.0, 1.15),
        ]
        if is_ned_pose_collision_free(q):
            valid_joints.append(q)
    return np.array(valid_joints, dtype=float)


if __name__ == "__main__":
    sample_joints = [-0.7854, 0.3504, 0.3554]
    pos = ned_forward_kinematics(sample_joints)
    print(f"Joints: {sample_joints}")
    print(f"FK TCP in ARM frame: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")