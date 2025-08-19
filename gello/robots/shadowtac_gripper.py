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
    CMD_GRIPPER_POSITION = 0x03
    CMD_DEBUG = 0x04
    CMD_LOG = 0x05

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
        self._min_position_mm = -25.0
        self._max_position_mm = 0.0
        self.last_position = 0

    def connect(self, port: str, baud_rate: int, timeout_s: float = 0.001) -> None:
        """Connects to the gripper over serial"""
        try:
            self.command_interface = CommandInterface(port, baud_rate=baud_rate, timeout_s=timeout_s)
            # Ensure that we are in a known state at start.
            self.command_interface.send_command(CommandID.CMD_GRIPPER_POSITION, struct.pack("<f", float(self._min_position_mm)))
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
        position = None
        if self.command_interface is not None:
            # Flush serial buffer
            while True:
                cmd, data = self.command_interface.receive_command()
                if cmd is None:
                    break
            # Trigger a position read from the sensor
            self.command_interface.send_command(CommandID.CMD_GRIPPER_POSITION, struct.pack(""))
            tries = 0
            # Wait for the response
            while tries < 5:
                cmd, data = self.command_interface.receive_command()
                if cmd is CommandID.CMD_GRIPPER_POSITION and data is not None and len(data) > 0:
                    position_mm = struct.unpack("f", data)[0]
                    position = ((self._max_position - self._min_position) / (self._max_position_mm - self._min_position_mm)) * position_mm - self._min_position_mm
                    self.last_position = position
                    break
                tries += 1
        return position if position is not None else self.last_position

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
        # _clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        # _clip_for = clip_val(self._min_force, force, self._max_force)

        success = False
        if self.command_interface is not None:
            # Get latest state 
            state = self._get_gripper_status()
            if state is not None and state is GripperState.IDLE:
                clip_pos_mm = ((self._max_position_mm - self._min_position_mm) / (self._max_position - self._min_position)) * clip_pos + self._min_position_mm
                self.command_interface.send_command(CommandID.CMD_GRIPPER_POSITION, struct.pack("<f", float(clip_pos_mm)))
                success = True
        
        return success, clip_pos

    def _get_gripper_status(self) -> Optional[GripperState]:
        if self.command_interface is not None:
            cmd, data = self.command_interface.receive_command()
            if cmd is CommandID.CMD_GRIPPER_STATE and data is not None:
                current_state = int.from_bytes(data, byteorder="little", signed=False)
                return GripperState(current_state)

        return None

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
