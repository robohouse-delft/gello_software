from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import toml
# import tyro
import glob
import sys 
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gello.dynamixel.driver import DynamixelDriver

MENAGERIE_ROOT: Path = Path(__file__).parent / "third_party" / "mujoco_menagerie"


@dataclass
class Args:
    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    start_joints: Tuple[float, ...] = (0, 0, 0, 0, 0, 0)
    """The joint angles that the GELLO is placed in at (in radians)."""

    joint_signs: Tuple[float, ...] = (1, 1, -1, 1, 1, 1)
    """The joint angles that the GELLO is placed in at (in radians)."""

    gripper: bool = True
    """Whether or not the gripper is attached."""

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert (
                j == -1 or j == 1
            ), f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_robot_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints


def get_config(config) -> None:
    env_config = config["env"]
    robot_config = config["robot"]
    usb_ports = glob.glob("/dev/serial/by-id/*")
    print(f"Found {len(usb_ports)} ports")
    port = None
    if len(usb_ports) > 0:
        port = usb_ports[0]
        print(f"using port {port}")
    else:
        raise ValueError(
            "No gello port found, please specify one or plug in gello"
        )

    start_joints = np.deg2rad(env_config["start_joints_deg"])
    num_robot_joints = len(start_joints)
    num_joints = num_robot_joints if robot_config["gripper"] == "none" else num_robot_joints + 1
    joint_ids = list(range(1, num_joints + 1))
    driver = DynamixelDriver(joint_ids, port=port, baudrate=57600)

    # assume that the joint state shouold be args.start_joints
    # find the offset, which is a multiple of np.pi/2 that minimizes the error between the current joint state and args.start_joints
    # this is done by brute force, we seach in a range of +/- 8pi

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = env_config["joint_signs"][index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = start_joints[index]
        return np.abs(joint_i - start_i)

    for _ in range(10):
        driver.get_joints()  # warmup

    for _ in range(1):
        best_offsets = []
        curr_joints = driver.get_joints()
        for i in range(num_robot_joints):
            best_offset = 0
            best_error = 1e6
            for offset in np.linspace(
                -8 * np.pi, 8 * np.pi, 8 * 4 + 1
            ):  # intervals of pi/2
                error = get_error(offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            best_offsets.append(best_offset)
        print()
        print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
        print(
            "best offsets function of pi: ["
            + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in best_offsets])
            + " ]",
        )
        if robot_config["gripper"] != "none":
            print(
                "gripper open (degrees)       ",
                np.rad2deg(driver.get_joints()[-1]) - 0.2,
            )
            print(
                "gripper close (degrees)      ",
                np.rad2deg(driver.get_joints()[-1]) - 42,
            )


def main(config) -> None:
    get_config(config)


if __name__ == "__main__":
    config = toml.load("./config.toml")
    main(config)
