import datetime
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from PIL import Image as PILImage
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from scripts.gello_to_lerobot_opt import GELLO_FEATURES, to_lerobot_frame

GELLO_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (7,), "names": ["joint_positions"]},
    "observation.joint_vel": {"dtype": "float32", "shape": (7,), "names": ["joint_velocities"]},
    "observation.ee_pose": {"dtype": "float32", "shape": (7,), "names": ["ee_pos_quat"]},
    "observation.images.wrist.rgb": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.wrist.depth": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.base.rgb": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.base.depth": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["joint_commands"]},
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
    frame["observation.joint_vel"] = step_data["joint_velocities"].astype(np.float32)
    frame["observation.ee_pose"] = step_data["ee_pos_quat"].astype(np.float32)
    frame["action"] = step_data["control"].astype(np.float32)

    # Handle images
    for key in list(step_data.keys()):
        if "rgb" in key or "depth" in key:
            cam_name, img_type = key.split("_")[0], key.split("_")[1]
            new_key = f"observation.images.{cam_name}.{img_type}"
            img = step_data[key].astype("uint8")
            if "depth" in key:
                img = depth_to_rgb(img)
            # Resize image.
            img = PILImage.fromarray(img).resize((256, 256), PILImage.Resampling.BICUBIC)
            frame[new_key] = np.array(img)
    return frame


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


class LeRobotDatasetRecorder:
    def __init__(self, repo_name: str, fps: int):
        self.dataset = LeRobotDataset.create(
            repo_id=repo_name,
            fps=fps,
            features=GELLO_FEATURES,
            image_writer_threads=10,
            image_writer_processes=5,
        )

    def add_frame(
            self,
            task: str,
            timestamp: datetime.datetime,
            obs: Dict[str, np.ndarray],
            action: np.ndarray) -> None:
        obs["control"] = action # add action to obs
        frame = to_lerobot_frame(obs)
        self.dataset.add_frame(frame, task, timestamp.timestamp())
    
    def save_episode(self) -> None:
        self.dataset.save_episode()