import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage
import tqdm
from lerobot.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently, calculate_episode_data_index, get_default_encoding
from lerobot.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

GELLO_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_positions"],
    },
    "observation.joint_vel": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_velocities"],
    },
    "observation.ee_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["ee_pos_quat"],
    },
    # "observation.images": {
    #     "dtype": "video",
    #     "shape": (640, 480, 3),
    #     "names": ["frames"],
    # },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_commands"],
    },
}

def load_from_raw(raw_dir, out_dir, fps, video, debug):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    episode_paths = list(raw_dir.glob("*"))
    episode_paths = [x for x in episode_paths if x.is_dir()]
    episode_paths = sorted(episode_paths)

    dataset_idx = 0
    episode_dicts = []
    episode_data_index = {}
    episode_data_index["from"] = []
    episode_data_index["to"] = []
    for episode_idx, episode_path in tqdm.tqdm(enumerate(episode_paths)):
        step_paths = list(episode_path.glob("*.pkl"))
        step_paths = sorted(step_paths)

        episode_dict = {}
        with open(step_paths[0], "rb") as f:
            for key in pickle.load(f):
                episode_dict[key] = []

        for key in ["frame_index", "episode_index", "index", "timestamp", "next.done"]:
            episode_dict[key] = []

        episode_start_idx = dataset_idx
        for step_idx, step_path in enumerate(step_paths):
            with open(step_path, "rb") as f:
                step_dict = pickle.load(f)
                for key in step_dict:
                    episode_dict[key].append(step_dict[key])

            episode_dict["frame_index"].append(step_idx)
            episode_dict["episode_index"].append(episode_idx)
            episode_dict["index"].append(step_idx)
            episode_dict["timestamp"].append(step_idx / fps)

            # assume all demonstrations are successful at the last step.
            episode_dict["next.done"].append(len(step_paths) - 1 == step_idx)
            # episode_dict["next.success"] = episode_dict["next.done"]

            dataset_idx += 1

        episode_data_index["from"].append(episode_start_idx)
        episode_data_index["to"].append(dataset_idx)

        # rename the keys to match the expected format

        episode_dict["action"] = episode_dict["control"]
        episode_dict.pop("control")

        episode_dict["observation.joint_vel"] = episode_dict["joint_velocities"]
        episode_dict.pop("joint_velocities")
        episode_dict["observation.ee_pose"] = episode_dict["ee_pos_quat"]
        episode_dict.pop("ee_pos_quat")


        # create spectrogram_img
        # episode_dict["spectogram_rgb"] = episode_dict["mic_spectrogram"]
        # episode_dict.pop("mic_spectrogram")

        # episode_dict["base-cropped_rgb"] = episode_dict["base_rgb_cropped"]
        # episode_dict.pop("base_rgb_cropped")

        for key in list(episode_dict.keys()):
            if "rgb" in key or "depth" in key:
                # parse the key
                cam_name, img_type = key.split("_")[0], key.split("_")[1]
                episode_dict[f"observation.images.{cam_name}.{img_type}"] = episode_dict[key]

                # drop the original key
                episode_dict.pop(key)

      
        # convert to the desired formats
        for key in episode_dict:
            if "rgb" in key or "depth" in key:
                # convert to uint8
                episode_dict[key] = [x.astype("uint8") for x in episode_dict[key]]
                if "depth" in key:
                    episode_dict[key] = [x[..., 0] for x in episode_dict[key]]

                # resize to 256x256
                from PIL import Image
                import numpy as np

                def resize(np_array):
                    img = Image.fromarray(np_array)
                    img = img.resize((256,256),Image.Resampling.BICUBIC)
                    img = np.array(img)
                    return img 
                episode_dict[key] = [resize(x) for x in episode_dict[key]]

                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = out_dir / "tmp_images"
                    save_images_concurrently(episode_dict[key], tmp_imgs_dir)
                    # encode images to a mp4 video
                    fname = f"{key}_episode_{episode_idx:06d}.mp4"
                    video_path = out_dir /  fname
                    encode_video_frames(tmp_imgs_dir, video_path, fps)

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    episode_dict[key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps}
                        for i in range(len(episode_dict[key]))
                    ]

                else:
                    episode_dict[key] = [PILImage.fromarray(x) for x in episode_dict[key]]

            else:
                episode_dict[key] = torch.tensor(episode_dict[key])

        episode_dicts.append(episode_dict)
        # state = joint_positions + gripper_position
        episode_dict["observation.state"] = torch.cat(
            [episode_dict["joint_positions"],
             # episode_dict["fingertips"],
             # episode_dict["accelerometer"].unsqueeze(1),
              ]
,
                dim=1
        )

        episode_dict.pop("joint_positions")
        episode_dict.pop("gripper_position")

        # for now, drop all depth images
        # for key in list(episode_dict.keys()):
        #     if "depth" in key:
        #         episode_dict.pop(key)

        # remove unused data 
        # episode_dict.pop("mic_frame")
        # episode_dict.pop("switches")
        # episode_dict.pop("wrench")

        if debug:
            break

    data_dict = concatenate_episodes(episode_dicts)
    return data_dict, episode_data_index


def to_hf_dataset(data_dict, video):
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    # features["next.success"] = Value(dtype="bool", id=None)

    features["observation.ee_pose"] = Sequence(
        length=data_dict["observation.ee_pose"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["observation.joint_vel"] = Sequence(
        length=data_dict["observation.joint_vel"].shape[1], feature=Value(dtype="float32", id=None)
    )
    #features["wrench"] = Sequence(Value(dtype="float32", id=None), length=6)
    # features["fingertips"] = Sequence(Value(dtype="float32", id=None), length=32)
    # features["accelerometer"] = Value(dtype="float32", id=None)


    # check if all keys in data dict are in feature dict

    for key in data_dict.keys():
        if key not in features.keys():
            print(f"key {key} was not found in HF Dataset feature list")

    for key in features.keys():
        if key not in data_dict.keys():
            print(f"key {key} in the HF dataset features was not found in the episode dict ")

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    if fps is None:
        fps = 10

    data_dict, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "fps": fps,
        "video": video,
    }

    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info

if __name__ == "__main__":
    repo_name = "ur5e_gello_cube_v1"
    task_name = "pick and place black cube into white bowl"
    fps = 100
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(Path("../data/gello"), Path("../lerobot_data"), fps=fps, video=False)

    dataset_path = Path.home() / ".cache/huggingface/lerobot/" / repo_name
    if dataset_path.exists():
        print(f"Deleting existing dataset folder: {dataset_path}")
        shutil.rmtree(dataset_path)
    
    print(hf_dataset)

    # Create dataset instance
    print("\nCreating dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        fps=fps,
        features=GELLO_FEATURES,
        tolerance_s=0.02,
        use_videos=False
    )

    print("Camera keys:", dataset.meta.camera_keys)
    episodes = range(len(episode_data_index["from"]))
    for ep_idx in episodes:
        from_idx = episode_data_index["from"][ep_idx].item()
        to_idx = episode_data_index["to"][ep_idx].item()
        num_frames = to_idx - from_idx

        for frame_idx in range(num_frames):
            i = from_idx + frame_idx
            frame_data = hf_dataset[i]

            # video_frame = generate_test_video_frame(
            #     width=640, height=480, frame_idx=frame_idx
            # )  # dummy images

            frame = {
                key: frame_data[key].numpy().astype(np.float32)
                for key in [
                    "observation.state",
                    "observation.joint_vel",
                    "observation.ee_pose",
                    "action"
                ]
            }
            # frame["observation.images"] = np.array(video_frame)
            # frame["timestamp"] = frame_data["timestamp"]

            dataset.add_frame(frame, task=task_name)

        dataset.save_episode()

    ds_meta = LeRobotDatasetMetadata(repo_name)

    # By instantiating just this class, you can quickly access useful information about the content and the
    # structure of the dataset without downloading the actual data yet (only metadata files — which are
    # lightweight).
    print(f"Total number of episodes: {ds_meta.total_episodes}")
    print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
    print(f"Frames per second used during data collection: {ds_meta.fps}")
    print(f"Robot type: {ds_meta.robot_type}")
    print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

    print("Tasks:")
    print(ds_meta.tasks)
    print("Features:")
    print(ds_meta.features)

    # You can also get a short summary by simply printing the object:
    print(ds_meta)


