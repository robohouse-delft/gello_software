import sys
import serial
import struct
import threading
import time
from enum import Enum, IntEnum
from typing import Tuple, Optional

# Commands that are sent between the host and the microcontroller
class CommandID(IntEnum):
    CMD_GRIPPER_STATE = 0x01
    CMD_SLIP_SAFETY_MARGIN = 0x02
    CMD_OPEN_GRIPPER = 0x03
    CMD_CLOSE_GRIPPER = 0x04
    CMD_DEBUG = 0x05
    CMD_LOG = 0x06

class GripperState(IntEnum):
    IDLE = 0x01
    MOVING = 0x02
    GRASPING = 0x03

class CommandInterface:
    def __init__(self, port: str, baud_rate: int, timeout_s: float = 0.001):
        self.serial_port = serial.Serial(port, baudrate=baud_rate, timeout=timeout_s)
    
    def close(self):
        self.serial_port.close()

    def send_command(self, cmd: CommandID, data: bytes):
        packet = struct.pack("<BB", cmd.value, len(data)) + data
        self.serial_port.write(packet)

    def receive_command(self) -> tuple[CommandID | None, bytes | None]:
        header = self.serial_port.read(2)
        if len(header) == 0:
            return None, None  # Not enough data

        cmd_id, length = struct.unpack("<BB", header)
        data = self.serial_port.read(length)
        if len(data) < length:
            return None, None

        try:
            cmd = CommandID(cmd_id)
        except ValueError:
            cmd = cmd_id  # Unknown command

        return cmd, data


class ShadowtacGripper:
    """Communicates with the gripper directly, via serial connection to shadowtac gripper"""

    class State(Enum):
        OPEN = 0x01
        CLOSED = 0x02

    def __init__(self):
        """Constructor."""
        self.command_interface: Optional[CommandInterface] = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255
        self.state = ShadowtacGripper.State.OPEN

    def connect(self, port: str, baud_rate: int, timeout_s: float = 0.001) -> None:
        """Connects to the gripper over serial"""
        try:
            self.command_interface = CommandInterface(port, baud_rate=baud_rate, timeout_s=timeout_s)
            # Ensure that we are in a known state at start.
            self.command_interface.send_command(CommandID.CMD_OPEN_GRIPPER, struct.pack(""))
        except serial.SerialException as e:
            sys.exit(e) # type: ignore

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        assert self.command_interface is not None
        self.command_interface.close()

    def _reset(self):
        # TODO: add reset.
        pass

    def is_active(self):
        """Returns whether the gripper is active."""
        return True

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
        return 0 if self.state == ShadowtacGripper.State.OPEN else 255

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

        success = False
        if self.command_interface is not None:
            # Add 50 points for hysterisis to compensate for the spring loaded return to "open"..not always precise.
            if (clip_pos < self._min_position + 100) and self.state != ShadowtacGripper.State.OPEN:
                self.command_interface.send_command(CommandID.CMD_OPEN_GRIPPER, struct.pack(""))
                self.state = ShadowtacGripper.State.OPEN
                success = True
            elif (clip_pos > self._max_position - 100) and self.state != ShadowtacGripper.State.CLOSED:
                self.command_interface.send_command(CommandID.CMD_CLOSE_GRIPPER, struct.pack(""))
                self.state = ShadowtacGripper.State.CLOSED
                success = True
        
        return success, clip_pos


def main():
    # test open and closing the gripper
    gripper = ShadowtacGripper()
    gripper.connect(port="/dev/ttyACM0", baud_rate=115200)
    print(gripper.get_current_position())
    gripper.move(255, 255, 1)
    time.sleep(0.2)
    print(gripper.get_current_position())
    gripper.move(0, 255, 1)
    time.sleep(0.2)
    print(gripper.get_current_position())
    gripper.move(255, 255, 1)
    gripper.disconnect()


if __name__ == "__main__":
    main()
