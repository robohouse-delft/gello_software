"""Module to control Robotiq's grippers - tested with HAND-E.

Taken from https://github.com/githubuser0xFFFF/py_robotiq_gripper/blob/master/src/robotiq_gripper.py
"""

import socket
import threading
import time
from enum import Enum
from typing import OrderedDict, Tuple, Union


class ShadowtacGripper:
    """Communicates with the gripper directly, via scoket connection to shadowtac"""

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        """Constructor."""
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 10.0) -> None:
        """Connects to a gripper at the given address.

        :param hostname: Hostname or ip.
        :param port: Port.
        :param socket_timeout: Timeout for blocking socket operations.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert self.socket is not None
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        assert self.socket is not None
        self.socket.close()

    def _reset(self):
        # TODO: add reset.
        pass

    def activate(self, auto_calibrate: bool = True):
        """Resets the activation flag in the gripper, and sets it back to one, clearing previous fault flags.

        :param auto_calibrate: Whether to calibrate the minimum and maximum positions based on actual motion.

        The following code is executed in the corresponding script function
        def rq_activate(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_reset(gripper_socket)

                while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                    rq_reset(gripper_socket)
                    sync()
                end

                rq_set_var("ACT",1, gripper_socket)
            end
        end

        def rq_activate_and_wait(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_activate(gripper_socket)
                sleep(1.0)

                while(not rq_get_var("ACT", 1, gripper_socket) == 1 or not rq_get_var("STA", 1, gripper_socket) == 3):
                    sleep(0.1)
                end

                sleep(0.5)
            end
        end
        """
        if not self.is_active():
            # TODO:
            pass

        # auto-calibrate position range if desired
        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self):
        """Returns whether the gripper is active."""
        return False

    def get_min_position(self) -> int:
        """Returns the minimum position the gripper can reach (open position)."""
        return self._min_position

    def get_max_position(self) -> int:
        """Returns the maximum position the gripper can reach (closed position)."""
        return self._max_position

    def get_open_position(self) -> int:
        """Returns what is considered the open position for gripper (minimum position value)."""
        return self.get_min_position()

    def get_closed_position(self) -> int:
        """Returns what is considered the closed position for gripper (maximum position value)."""
        return self.get_max_position()

    def is_open(self):
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self):
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()

    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        return 0

    def auto_calibrate(self, log: bool = True) -> None:
        # TODO:

        if log:
            print(
                f"Gripper auto-calibrated to [{self.get_min_position()}, {self.get_max_position()}]"
            )

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """Sends commands to start moving towards the given position, with the specified speed and force.

        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with a bool indicating whether the action it was successfully sent, and an integer with
        the actual position that was requested, after being adjusted to the min/max calibrated range.
        """
        position = int(position)
        speed = int(speed)
        force = int(force)

        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))

        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        clip_for = clip_val(self._min_force, force, self._max_force)

        # TODO: moves to the given position with the given speed and force
        
        return False, clip_pos

    def move_and_wait_for_pos(
        self, position: int, speed: int, force: int
    ) -> Tuple[int, ObjectStatus]:  # noqa
        """Sends commands to start moving towards the given position, with the specified speed and force, and then waits for the move to complete.

        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with an integer representing the last position returned by the gripper after it notified
        that the move had completed, a status indicating how the move ended (see ObjectStatus enum for details). Note
        that it is possible that the position was not reached, if an object was detected during motion.
        """
        position = int(position)
        speed = int(speed)
        force = int(force)

        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for move.")

        # TODO: wait until the gripper acknowledges that it will try to go to the requested position

        # TODO: wait until not moving

        # TODO: report the actual position and the object status
        return (0, ShadowtacGripper.ObjectStatus.AT_DEST)


def main():
    # test open and closing the gripper
    gripper = ShadowtacGripper()
    gripper.connect(hostname="192.168.1.10", port=63352)
    # gripper.activate()
    print(gripper.get_current_position())
    gripper.move(20, 255, 1)
    time.sleep(0.2)
    print(gripper.get_current_position())
    gripper.move(65, 255, 1)
    time.sleep(0.2)
    print(gripper.get_current_position())
    gripper.move(20, 255, 1)
    gripper.disconnect()


if __name__ == "__main__":
    main()
