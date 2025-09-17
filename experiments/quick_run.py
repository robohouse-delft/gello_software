import atexit
import toml
import tyro
from dataclasses import dataclass
from multiprocessing import Process

from launch_nodes import launch_robot_server
from launch_camera_nodes import main as camera_main
from run_env import main as env_main


@dataclass
class Args:
    agent: str = "none"
    mock: bool = False
    display_data: bool = False
    verbose: bool = False

def start_robot_process(config):
    process = Process(target=launch_robot_server, args=(config, ))

    # Function to kill the child process
    def kill_child_process(process):
        print("Killing child process...")
        process.terminate()

    # Register the kill_child_process function to be called at exit
    atexit.register(kill_child_process, process)
    process.start()

def start_camera_process(config):
    process = Process(target=camera_main, args=(config["camera_server"], ))

    # Function to kill the child process
    def kill_child_process(process):
        print("Killing child process...")
        process.terminate()

    # Register the kill_child_process function to be called at exit
    atexit.register(kill_child_process, process)
    process.start()


def main(config, args: Args):
    # Run the camera and robot processes in child processes.
    start_camera_process(config)
    start_robot_process(config)

    # Run the main environment
    env_main(config, args)


if __name__ == "__main__":
    config = toml.load("./config.toml")
    args = tyro.cli(Args)
    main(config, args)
