import time
from typing import Any, Dict, Optional

import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.t_prev = time.monotonic()
        self.dt = 1.0 / rate
        self.slack_time = 0.001

    def sleep(self) -> None:
        # Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/precise_sleep.py
        t_current = time.monotonic()
        t_next = self.t_prev + self.dt
        t_wait = t_next - t_current
        if t_wait > 0:
            t_sleep = t_wait - self.slack_time
            if t_sleep > 0:
                time.sleep(t_sleep)
            while time.monotonic() < t_next:
                pass
        self.t_prev = time.monotonic()
        # while self.t_prev + self.dt > time.perf_counter():
        #     time.sleep(0.0001)
        # self.t_prev = time.perf_counter()



class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def close(self):
        self._robot.stop()
        for name, camera in self._camera_dict.items():
            camera.stop()

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (self._robot.num_dofs()), (
            f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        )
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = np.asarray(robot_obs["joint_positions"])
        observations["joint_velocities"] = np.asarray(robot_obs["joint_velocities"])
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()
