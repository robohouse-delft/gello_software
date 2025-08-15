import datetime
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scripts.gello_to_lerobot_opt import GELLO_FEATURES, to_lerobot_frame


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