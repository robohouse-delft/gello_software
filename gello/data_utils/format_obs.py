import datetime
import pickle
from pathlib import Path
from typing import Dict, Protocol
from abc import abstractmethod

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .keyboard_interface import init_keyboard_listener

GELLO_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (7,), "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
    # "observation.state": {"dtype": "float32", "shape": (7,), "names": ["joint_positions"]},
    # "observation.joint_vel": {"dtype": "float32", "shape": (7,), "names": ["joint_velocities"]},
    # "observation.ee_pose": {"dtype": "float32", "shape": (7,), "names": ["ee_pos_quat"]},
    "observation.images.wrist.rgb": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
    # "observation.images.wrist.depth": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.base.rgb": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
    # "observation.images.base.depth": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    # "action": {"dtype": "float32", "shape": (7,), "names": ["joint_commands"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
}

def depth_to_rgb(depth_img):
    """Convert single-channel depth image to 3-channel grayscale RGB."""
    if depth_img.ndim == 3 and depth_img.shape[-1] == 1:  # (H, W, 1)
        return np.repeat(depth_img, 3, axis=-1)
    elif depth_img.ndim == 3 and depth_img.shape[0] == 1:  # (1, H, W)
        return np.repeat(depth_img, 3, axis=0)
    else:
        raise ValueError(f"Unexpected depth shape: {depth_img.shape}")

def to_lerobot_frame(step_data: Dict) -> Dict:
    # Remap keys
    frame = {}
    frame["observation.state"] = step_data["joint_positions"].astype(np.float32)
    # frame["observation.joint_vel"] = step_data["joint_velocities"].astype(np.float32)
    # frame["observation.ee_pose"] = step_data["ee_pos_quat"].astype(np.float32)
    frame["action"] = step_data["control"].astype(np.float32)

    # Handle images
    for key in list(step_data.keys()):
        # if "rgb" in key or "depth" in key:
        if "rgb" in key:
            cam_name, img_type = key.split("_")[0], key.split("_")[1]
            new_key = f"observation.images.{cam_name}.{img_type}"
            img = step_data[key].astype("uint8")
            # if "depth" in key:
                # img = depth_to_rgb(img)
            # Resize image.
            # img = PILImage.fromarray(img).resize((256, 256), PILImage.Resampling.BICUBIC)
            frame[new_key] = np.array(img)
    return frame


class DatasetRecorder:
    def __init__(self):
        self.keyboard_listener, self.keyboard_events = init_keyboard_listener()

    def check_event(self, key) -> bool:
        return self.keyboard_events[key]
    
    def clear_events(self):
        for key in self.keyboard_events.keys():
            self.keyboard_events[key] = False
    
    def close(self):
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()


    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_frame(
        self,
        task: str,
        timestamp: float,
        obs: Dict[str, np.ndarray],
        action: np.ndarray) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def discard_episode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_episode(self) -> None:
        raise NotImplementedError
    

def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    obs["control"] = action  # add action to obs

    # make folder if it doesn't exist
    folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder / (timestamp.isoformat() + ".pkl")

    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)

class GelloDatasetRecorder(DatasetRecorder):
    def __init__(self, repo_name: str, fps: int):
        super().__init__()
        self.root_dir = Path(repo_name)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = None
    
    def start(self) -> None:
        data_timestamp = datetime.datetime.now()
        self.out_dir = self.root_dir.joinpath(data_timestamp.strftime("%m%d_%H%M%S"))
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def add_frame(
            self,
            task: str,
            timestamp: float,
            obs: Dict[str, np.ndarray],
            action: np.ndarray) -> None:
        data_timestamp = datetime.datetime.now()
        obs["control"] = action  # add action to obs

        # make folder if it doesn't exist
        if self.out_dir is not None:
            recorded_file = self.out_dir / (data_timestamp.isoformat() + ".pkl")
            with open(recorded_file, "wb") as f:
                pickle.dump(obs, f)
    
    def discard_episode(self) -> None:
        self.out_dir = None
    
    def save_episode(self) -> None:
        self.out_dir = None

class LeRobotDatasetRecorder(DatasetRecorder):
    def __init__(self, repo_name: str, fps: int):
        super().__init__()
        dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_name
        if dataset_path.exists():
            print(f"Using existing dataset folder: {dataset_path}")
            self.dataset = LeRobotDataset(repo_id=str(dataset_path), tolerance_s=0.003)
            self.dataset.start_image_writer(
                num_processes=0,
                num_threads=16,
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=repo_name,
                fps=fps,
                features=GELLO_FEATURES,
                image_writer_threads=16,
                image_writer_processes=0,
                tolerance_s=0.003,
            )
    
    def start(self) -> None:
        pass
            
    def add_frame(
            self,
            task: str,
            timestamp: float,
            obs: Dict[str, np.ndarray],
            action: np.ndarray) -> None:
        obs["control"] = action # add action to obs
        frame = to_lerobot_frame(obs)
        self.dataset.add_frame(frame, task, timestamp=timestamp)
    
    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()
    
    def save_episode(self) -> None:
        self.dataset.save_episode()