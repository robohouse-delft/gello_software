from typing import Dict, Union, List

import numpy as np

from gello.robots.robot import Robot


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
        self,
        robot_ip: str,
        state_feedback_hz: float,
        gripper: str,
        gripper_hostname: str,
        gripper_port: int,
        start_position: Union[List, np.ndarray],
        x_limits: Union[List, np.ndarray],
        y_limits: Union[List, np.ndarray],
        z_limits: Union[List, np.ndarray],
    ):
        import rtde_control
        from rtde_control import RTDEControlInterface as RTDEControl
        import rtde_receive
        self.freq_hz = 125.0
        self.state_feedback_hz = state_feedback_hz
        print(f"Init UR RTDE interface at {self.freq_hz} Hz with GELLO state feedback at {self.state_feedback_hz} Hz")
        try:
            self.robot = rtde_control.RTDEControlInterface(
                robot_ip, 125.0, RTDEControl.FLAG_USE_EXT_UR_CAP
            )
        except Exception as e:
            print(e)
            print(robot_ip)
            raise ConnectionError(e)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        self._use_gripper = True
        if gripper == "none":
            self._use_gripper = False
        elif gripper == "robotiq":
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=gripper_hostname, port=gripper_port)
            print(f"Robotiq gripper ({gripper_hostname}:{gripper_port}) connected")
        elif gripper == "shadowtac":
            from gello.robots.shadowtac_gripper import ShadowtacGripper

            self.gripper = ShadowtacGripper()
            self.gripper.connect(port=gripper_hostname, baud_rate=gripper_port)
            print(f"Shadowtac gripper ({gripper_hostname}:{gripper_port}) connected")
        else:
            raise ValueError("Invalid gripper type specified")

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.tcp_offset = self.robot.getTCPOffset()
        self.robot.moveJ(start_position)
        self.outside_workspace_limits = False

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.0
        acceleration = 0.0
        dt = 1.0 / self.state_feedback_hz
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        # Check limits end-effector workspace limits before commanding the robot
        if not self._is_within_limits(robot_joints):
            if not self.outside_workspace_limits:
                print(
                    "Robot end-effector has been commanded to be outside of the workspace limits. Move leader arm back to within workspace."
                )
            self.outside_workspace_limits = True
            return None
        elif self.outside_workspace_limits:
            print("Robot end-effector back inside the workspace")
            self.outside_workspace_limits = False

        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        if self._use_gripper:
            gripper_pos = joint_state[-1] * 255
            self.gripper.move(gripper_pos, 255, 10)
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }

    def _is_within_limits(self, q_vec: np.ndarray) -> bool:
        within_limits = True
        try:
            pose = self.robot.getForwardKinematics(q_vec, self.tcp_offset)
            if pose[0] < self.x_limits[0] or pose[0] > self.x_limits[1]:
                within_limits = False
            elif pose[1] < self.y_limits[0] or pose[1] > self.y_limits[1]:
                within_limits = False
            elif pose[2] < self.z_limits[0] or pose[2] > self.z_limits[1]:
                within_limits = False
        except RuntimeError as e:
            print(e)
            within_limits = False
        return within_limits


def main():
    robot_ip = "192.168.1.11"
    gripper_hostname = "192.168.0.4"
    gripper_port = 63352
    ur = URRobot(
        robot_ip,
        10.0,
        "none",
        gripper_hostname,
        gripper_port,
        start_position=[0.0, -90.0, -90.0, -90.0, 90.0, 0.0],
        x_limits=[0.0, 0.5],
        y_limits=[-0.3, 0.5],
        z_limits=[0.05, 0.7],
    )
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
