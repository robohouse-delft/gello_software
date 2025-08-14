import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
import tqdm

from lerobot.datasets.utils import hf_transform_to_torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# ---------- CONFIG ----------
GELLO_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (7,), "names": ["joint_positions"]},
    "observation.joint_vel": {"dtype": "float32", "shape": (7,), "names": ["joint_velocities"]},
    "observation.ee_pose": {"dtype": "float32", "shape": (7,), "names": ["ee_pos_quat"]},
    "observation.images.wrist.rgb": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
    "observation.images.wrist.depth": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
    "observation.images.base.rgb": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
    "observation.images.base.depth": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
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

def to_channels_last(img):
    if hasattr(img, "permute"):  # torch.Tensor
        return img.permute(1, 2, 0)
    elif isinstance(img, np.ndarray):
        return np.transpose(img, (1, 2, 0))
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

def process_episode(episode_path, dataset, fps, task_name):
    """Load one episode and save it directly to LeRobotDataset."""
    step_paths = sorted(episode_path.glob("*.pkl"))

    with open(step_paths[0], "rb") as f:
        first_step = pickle.load(f)
        keys = list(first_step.keys())

    for step_idx, step_path in enumerate(step_paths):
        with open(step_path, "rb") as f:
            step_data = pickle.load(f)

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
                # img = PILImage.fromarray(img).resize((256, 256), PILImage.Resampling.BICUBIC)
                frame[new_key] = np.array(img)

        dataset.add_frame(frame, task=task_name)

    dataset.save_episode()
    # del step_data, frame
    # torch.cuda.empty_cache()

def convert_to_lerobot(raw_dir, out_dir, repo_name, task_name, fps=30):
    dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_name
    if dataset_path.exists():
        print(f"Deleting existing dataset folder: {dataset_path}")
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        fps=fps,
        features=GELLO_FEATURES,
        tolerance_s=0.03,
        use_videos=True,
    )

    episode_paths = sorted([x for x in Path(raw_dir).glob("*") if x.is_dir()])
    for ep_path in tqdm.tqdm(episode_paths, desc="Processing episodes"):
        process_episode(ep_path, dataset, fps, task_name)

    print("Conversion complete!")
    ds_meta = LeRobotDatasetMetadata(repo_name)
    print(ds_meta)

if __name__ == "__main__":
    repo_name = "ur5e_gello_cube_v1"
    task_name = "pick and place black cube into white bowl"
    gello_dir = Path.home() / "dev/gello_software/data/gello"
    videos_dir = Path.home() / "dev/gello_software/data/videos/ur5e_gello_cube_v1"

    convert_to_lerobot(gello_dir, videos_dir, repo_name, task_name, fps=30)
