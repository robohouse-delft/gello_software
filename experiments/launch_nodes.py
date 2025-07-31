from dataclasses import dataclass
from pathlib import Path

import toml
# import tyro

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "xarm"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.10"
    no_gripper: bool = False


def launch_robot_server(config):
    port = config["node_port"]
    if config["type"] == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=config["node_hostname"]
        )
        server.serve()
    elif config["type"] == "sim_panda":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=config["node_hostname"]
        )
        server.serve()
    elif config["type"] == "sim_xarm":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=config["node_hostname"]
        )
        server.serve()

    else:
        if config["type"] == "xarm":
            from gello.robots.xarm_robot import XArmRobot

            robot = XArmRobot(ip=config["hostname"])
        elif config["type"] == "ur":
            from gello.robots.ur import URRobot

            robot = URRobot(robot_ip=config["hostname"], no_gripper=config["no_gripper"])
        elif config["type"] == "panda":
            from gello.robots.panda import PandaRobot

            robot = PandaRobot(robot_ip=config["hostname"])
        elif config["type"] == "bimanual_ur":
            from gello.robots.ur import URRobot

            # IP for the bimanual robot setup is hardcoded
            _robot_l = URRobot(robot_ip="192.168.2.10")
            _robot_r = URRobot(robot_ip="192.168.1.10")
            robot = BimanualRobot(_robot_l, _robot_r)
        elif config["type"] == "none" or config["type"] == "print":
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot {config['type']} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
            )
        server = ZMQServerRobot(robot, port=port, host=config["node_hostname"])
        print(f"Starting robot server on port {port}")
        server.serve()


def main(config):
    launch_robot_server(config)


if __name__ == "__main__":
    config = toml.load("./config.toml")
    main(config["robot"])
