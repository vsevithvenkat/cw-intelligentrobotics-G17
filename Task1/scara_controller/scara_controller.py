# scara_controller.py
from controller import Supervisor
import numpy as np
from scara_fk import scara_forward_kinematics, sample_scara_collision_free

ACTIVE_MOTOR_NAMES = ["base_arm_motor", "arm_motor", "shaft_linear_motor"]
FIXED_MOTOR_NAMES = ["shaft_rotation_motor"]


class ScaraController:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        # 1. Initialize active motors and sensors
        self.active_motors = []
        self.sensors = []
        for name in ACTIVE_MOTOR_NAMES:
            motor = self.robot.getDevice(name)
            if motor is None:
                raise RuntimeError(f"ERROR: Could not find motor '{name}' on SCARA.")
            self.active_motors.append(motor)

            sensor = motor.getPositionSensor()
            if sensor is not None:
                sensor.enable(self.timestep)
            self.sensors.append(sensor)

        # 2. Fix ignored shaft rotation motor to 0.0 rad
        self.fixed_motors = []
        for name in FIXED_MOTOR_NAMES:
            m = self.robot.getDevice(name)
            if m is not None:
                m.setPosition(0.0)
                self.fixed_motors.append(m)

        # 3. Locate ARM and TCP nodes
        self.arm = self.robot.getFromDef("ARM")
        if self.arm is None:
            self.arm = self.robot.getSelf()

        self.tcp = self.robot.getFromDef("TCP")
        if self.tcp is None:
            for field_name in ["tcpSlot", "handSlot"]:
                slot = self.arm.getField(field_name)
                if slot is not None and slot.getCount() > 0:
                    self.tcp = slot.getMFNode(0)
                    break

        if self.tcp is None:
            raise RuntimeError("ERROR: Could not find DEF TCP or tcpSlot/handSlot.")

        print("SCARA TCP node found successfully.")

    def move_to_joints_and_settle(self, target_joints, tol=1e-4, vel_tol=1e-5, max_steps=2000):
        for motor, val in zip(self.active_motors, target_joints):
            motor.setPosition(float(val))

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

                # Wait until both close to target AND completely stopped moving
                if max_err < tol and max_delta < vel_tol:
                    settled_count += 1
                    if settled_count >= 15:
                        break
                else:
                    settled_count = 0

        return self.get_supervisor_tcp()

    def get_supervisor_tcp(self):
        pose = self.tcp.getPose(self.arm)
        return np.array([pose[3], pose[7], pose[11]], dtype=float)


if __name__ == "__main__":
    controller = ScaraController()
    rng = np.random.default_rng(42)

    diagnostic_tests = [
        [0.0, 0.0, 0.0],       # Home pose
        [0.4, 0.0, 0.0],       # Base rotation only
        [0.0, 0.4, 0.0],       # Elbow rotation only (q2 > 0)
        [0.0, 0.0, -0.15],     # Linear shaft displacement only
    ]

    # 20 random collision-free elbow-up test poses (Option 1 Assumption):
    # q1 in [-0.73, 0.73], q2 in [0.0, 0.83], q4 in [-0.170, 0.0]
    random_tests = sample_scara_collision_free(20, rng).tolist()

    all_tests = diagnostic_tests + random_tests
    errors = []

    print(f"{'Sample':<6} | {'Joints [q1, q2, q4]':<26} | {'FK TCP [x, y, z]':<28} | {'Sim TCP [x, y, z]':<28} | {'Error (mm)'}")
    print("-" * 110)

    for idx, joints in enumerate(all_tests):
        sim_tcp = controller.move_to_joints_and_settle(joints)
        fk_tcp = scara_forward_kinematics(joints)
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