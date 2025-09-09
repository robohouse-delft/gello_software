import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import toml

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.env import RobotEnv, Rate
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.data_utils.format_obs import DataController, LeRobotDatasetRecorder, GelloDatasetRecorder
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    no_gripper: bool = False
    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False


def main(config):
    env_config = config["env"]
    robot_config = config["robot"]
    if env_config["mock"]:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            "wrist": ZMQClientCamera(port=env_config["wrist_camera_port"], host=config["camera_server"]["hostname"]),
            "base": ZMQClientCamera(port=env_config["base_camera_port"], host=config["camera_server"]["hostname"]),
        }
        robot_client = ZMQClientRobot(port=env_config["robot_port"], host=env_config["hostname"])
    env = RobotEnv(robot_client, control_rate_hz=125, camera_dict=camera_clients)

    if env_config["display_data"]:
        _init_rerun(("recording" if env_config["agent"] == "gello" else "inference"))

    if env_config["bimanual"]:
        if env_config["agent"] == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif env_config["agent"] == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=env_config["robot_type"], which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=env_config["robot_type"], which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)
            # raise NotImplementedError
        elif env_config["agent"] == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=env_config["robot_type"], device_path=left_path, verbose=env_config["verbose"]
            )
            right_agent = SpacemouseAgent(
                robot_type=env_config["robot_type"],
                device_path=right_path,
                verbose=env_config["verbose"],
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {env_config['agent']}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = (
            np.deg2rad([0, -90, -90, -90, 90, 0, 0])
            if robot_config["gripper"] != "none"
            else np.deg2rad([0, -90, -90, -90, 90, 0])
        )
        reset_joints_right = (
            np.deg2rad([0, -90, 90, -90, -90, 0, 0])
            if robot_config["gripper"] != "none"
            else np.deg2rad([0, -90, -90, -90, 90, 0])
        )
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if env_config["agent"] == "gello":
            gello_port = None
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                if len(usb_ports) > 1:
                    gello_port = usb_ports[1] if "teensy" in gello_port else gello_port
                print(f"using port {gello_port}")
            else:
                raise ValueError(
                    "No gello port found, please specify one or plug in gello"
                )
            if env_config["start_joints_deg"] is None:
                reset_joints = np.deg2rad(
                    [0, -90, 90, -90, -90, 0, 0]
                )  # Change this to your own reset joints
                if robot_config["gripper"] == "none":
                    reset_joints = reset_joints[:-1]
            else:
                reset_joints = np.asarray(np.deg2rad(env_config["start_joints_deg"]))
            gripper_config = None if robot_config["gripper"] == "none" else env_config["gripper_config"]
            start_joints = None
            if gripper_config is not None:
                reset_joints = reset_joints + [0]
                start_joints = np.deg2rad(env_config["start_joints_deg"] + [0.0])
            agent = GelloAgent(
                port=gello_port,
                joint_ids=env_config["joint_ids"],
                joint_signs=env_config["joint_signs"],
                joint_offsets=env_config["joint_offsets"],
                gripper_config=gripper_config,
                start_joints=start_joints
            )
            curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif env_config["agent"] == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=env_config["robot_type"], which_hand="l")
        elif env_config["agent"] == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=env_config["robot_type"], verbose=env_config["verbose"])
        elif env_config["agent"] == "dummy" or env_config["agent"] == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif env_config["agent"] == "lerobot_replay":
            from gello.agents.lerobot_agent import LeRobotReplayAgent

            agent = LeRobotReplayAgent(config["lerobot"]["dataset_url"], episode_idx=config["lerobot"]["episode_idx"])
        elif env_config["agent"] == "lerobot_policy":
            from gello.agents.lerobot_agent import LeRobotAgent, load_act_policy, load_smolvla_policy, load_diffusion_policy

            policy = None
            match config["lerobot"]["policy"]:
                case "act":
                    policy = load_act_policy(config["lerobot"]["checkpoint_path"])
                case "smolvla":
                    policy = load_smolvla_policy(config["lerobot"]["checkpoint_path"])
                case "diffusion":
                    policy = load_diffusion_policy(config["lerobot"]["checkpoint_path"])
                case _:
                    raise ValueError("Invalid policy name")
                
            agent = LeRobotAgent(policy, config["lerobot"]["task"])
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    recorder = DataController()
    obs = env.get_obs()
    if env_config["agent"] == "gello":
        print("Going to start position")
        start_pos = np.asarray(agent.act(env.get_obs()))
        joints = obs["joint_positions"]

        abs_deltas = np.abs(start_pos - joints)
        id_max_joint_delta = np.argmax(abs_deltas)

        max_joint_delta = 0.8
        if abs_deltas[id_max_joint_delta] > max_joint_delta:
            id_mask = abs_deltas > max_joint_delta
            print()
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                ids,
                abs_deltas[id_mask],
                start_pos[id_mask],
                joints[id_mask],
            ):
                print(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            return

        print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
        assert len(start_pos) == len(joints), (
            f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"
        )

        max_delta = 0.05
        for _ in range(25):
            obs = env.get_obs()
            command_joints = agent.act(obs)
            current_joints = obs["joint_positions"]
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            env.step(current_joints + delta)

        obs = env.get_obs()
        joints = obs["joint_positions"]
        action = agent.act(obs)
        if (action - joints > 0.5).any():
            print("Action is too big")

            # print which joints are too big
            joint_index = np.where(action - joints > 0.8)
            for j in joint_index:
                print(
                    f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                )
            exit()

        # Recording datasets
        if config["lerobot"]["dataset_type"] == "gello":
            recorder = GelloDatasetRecorder((Path(env_config["data_dir"]).expanduser() / config["lerobot"]["dataset_url"]).as_posix(), env_config["freq_hz"])
        else:
            recorder = LeRobotDatasetRecorder(config["lerobot"]["dataset_url"], env_config["freq_hz"])


    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    recording_start_timestamp = time.perf_counter()
    loop_rate_hz = Rate(env_config["freq_hz"])
    # prev_timestamp = 0
    recording = False
    obs_timestamp = 0
    while True:
        # Act and observe
        # prev_timestamp = obs_timestamp
        obs_timestamp = time.perf_counter()
        # print(f"loop dt: {obs_timestamp - prev_timestamp}")
        action = agent.act(obs)
        obs = env.step(action)
        # Check events from user keyboard and act appropriately
        if recorder.check_event("start_recording"):
            print("Start episode recording")
            recording_start_timestamp = time.perf_counter()
            recording = True
            recorder.start()
        elif recorder.check_event("stop_recording"):
            print("Saving episode...", end="")
            recorder.save_episode()
            print("done")
        elif recorder.check_event("discard_recording"):
            print("Discard recording")
            recorder.discard_episode()
        elif recorder.check_event("quit"):
            print("Quit loop!")
            break
        if recording:
            timestamp = obs_timestamp - recording_start_timestamp
            recorder.add_frame(config["lerobot"]["task"], timestamp, obs, action)
        if env_config["display_data"]:
            log_rerun_data(obs, dict(np.ndenumerate(action)))
        recorder.clear_events()
        # Sleep to control the loop rate
        loop_rate_hz.sleep()
    
    # Close the environment
    agent.stop()
    env.close()
    if recorder is not None:
        recorder.close()

    print("Program end!")

if __name__ == "__main__":
    config = toml.load("./config.toml")
    main(config)
