# ned_controller.py
from controller import Supervisor
import numpy as np
from ned_fk import ned_forward_kinematics, sample_ned_collision_free


class NedController:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # 1. Initialize active motors (from ned_simple.py)
        self.waist = self.robot.getDevice("joint_1")
        self.shoulder = self.robot.getDevice("joint_2")
        self.elbow = self.robot.getDevice("joint_3")
        self.active_motors = [self.waist, self.shoulder, self.elbow]

        # Enable position sensors to detect when the arm has physically settled
        self.sensors = []
        for motor in self.active_motors:
            sensor = motor.getPositionSensor()
            if sensor is not None:
                sensor.enable(self.timestep)
            self.sensors.append(sensor)

        # 2. Fix all wrist joints to 0.0 (Task 1 specification)
        self.fixed_motors = []
        for name in ["joint_4", "joint_5", "joint_6"]:
            m = self.robot.getDevice(name)
            if m is not None:
                m.setPosition(0.0)
                self.fixed_motors.append(m)

        # 3. Locate ARM and TCP nodes (exact logic from tcp_pose.py)
        self.arm = self.robot.getFromDef("ARM")
        if self.arm is None:
            self.arm = self.robot.getSelf()

        self.tcp = self.robot.getFromDef("TCP")
        if self.tcp is None:
            print("Could not find global DEF TCP. Trying ARM.tcpSlot instead...")
            tcp_slot = self.arm.getField("tcpSlot")
            if tcp_slot is None or tcp_slot.getCount() == 0:
                raise RuntimeError("ERROR: ARM.tcpSlot is missing or empty.")
            self.tcp = tcp_slot.getMFNode(0)

        print("Ned TCP node found successfully.")

    def move_to_joints_and_settle(self, target_joints, tol=1e-4, vel_tol=1e-5, max_steps=2000):
        """
        Commands [joint_1, joint_2, joint_3], waits for the arm to settle,
        and returns the measured TCP position in the ARM frame.
        """
        for motor, angle in zip(self.active_motors, target_joints):
            motor.setPosition(float(angle))

        for motor in self.fixed_motors:
            motor.setPosition(0.0)

        prev_pos = None
        settled_count = 0

        for _ in range(max_steps):
            if self.robot.step(self.timestep) == -1:
                break

            if all(s is not None for s in self.sensors):
                current = [s.getValue() for s in self.sensors]
                max_err = max(abs(c - t) for c, t in zip(current, target_joints))

                if prev_pos is not None:
                    max_delta = max(abs(c - p) for c, p in zip(current, prev_pos))
                else:
                    max_delta = float("inf")
                prev_pos = current

                if max_err < tol and max_delta < vel_tol:
                    settled_count += 1
                    if settled_count >= 15:
                        break
                else:
                    settled_count = 0

        return self.get_supervisor_tcp()

    def get_supervisor_tcp(self):
        """
        Obtains the measured TCP position relative to the ARM frame
        using the exact 4x4 homogeneous transform from tcp_pose.py.
        """
        pose = self.tcp.getPose(self.arm)
        return np.array([pose[3], pose[7], pose[11]], dtype=float)


if __name__ == "__main__":
    controller = NedController()
    rng = np.random.default_rng(42)

    # Diagnostic cardinal poses + ned_simple example pose
    diagnostic_tests = [
        [0.0, 0.0, 0.0],            # Zero/Home configuration
        [-0.7854, 0.3504, 0.3554],  # Pose from ned_simple.py
        [0.5, 0.0, 0.0],            # Waist rotation only
        [0.0, 0.4, 0.0],            # Shoulder pitch only
        [0.0, 0.0, 0.4],            # Elbow pitch only (elbow-up: q3 > 0)
    ]

    # 20 random collision-free elbow-up poses (Option 1 Assumption)
    random_tests = sample_ned_collision_free(20, rng).tolist()

    all_tests = diagnostic_tests + random_tests
    errors = []

    print(f"{'Sample':<6} | {'Joints [q1, q2, q3]':<26} | {'FK TCP [x, y, z]':<28} | {'Sim TCP [x, y, z]':<28} | {'Error (mm)'}")
    print("-" * 110)

    for idx, joints in enumerate(all_tests):
        sim_tcp = controller.move_to_joints_and_settle(joints)
        fk_tcp = ned_forward_kinematics(joints)
        err_m = np.linalg.norm(fk_tcp - sim_tcp)
        errors.append(err_m)

        j_str = f"[{joints[0]:6.3f}, {joints[1]:6.3f}, {joints[2]:6.3f}]"
        fk_str = f"[{fk_tcp[0]:7.4f}, {fk_tcp[1]:7.4f}, {fk_tcp[2]:7.4f}]"
        sim_str = f"[{sim_tcp[0]:7.4f}, {sim_tcp[1]:7.4f}, {sim_tcp[2]:7.4f}]"
        print(f"{idx:<6} | {j_str:<26} | {fk_str:<28} | {sim_str:<28} | {err_m * 1000:.4f} mm")

    errors = np.array(errors)
    print("-" * 110)
    print(f"Mean Position Error: {np.mean(errors) * 1000:.4f} mm ({np.mean(errors):.6f} m)")
    print(f"Std  Position Error: {np.std(errors) * 1000:.4f} mm ({np.std(errors):.6f} m)")
    print(f"Max  Position Error: {np.max(errors) * 1000:.4f} mm ({np.max(errors):.6f} m)")