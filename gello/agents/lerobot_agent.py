import time
import numpy as np
from typing import Dict
import torch
from torchvision.transforms import Resize, InterpolationMode
from PIL import Image as PILImage
from gello.agents.agent import Agent
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait

def load_act_policy(checkpoint_path: str) -> ACTPolicy:
    # Load the checkpoint
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    return policy

class LeRobotAgent(Agent):
    def __init__(self, policy: ACTPolicy) -> None:
        self.policy = policy
        self.image_resizer = Resize((256, 256), interpolation=InterpolationMode.BICUBIC)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        takes in a gello observation, turns it into a Lerobot observation and returns the action in gello format
        """

        base_rgb_img = obs["base_rgb"]
        base_depth_img = obs["base_depth"]
        wrist_rgb_img = obs["wrist_rgb"]
        wrist_depth_img = obs["wrist_depth"]
        joint_positions = obs["joint_positions"]

        # Resize images
        # base_rgb_img = Resize(base_rgb_img).resize((256, 256), PILImage.Resampling.BICUBIC)
        # base_depth_img = PILImage.fromarray(base_depth_img).resize((256, 256), PILImage.Resampling.BICUBIC)
        # wrist_rgb_img = PILImage.fromarray(wrist_rgb_img).resize((256, 256), PILImage.Resampling.BICUBIC)
        # wrist_depth_img = PILImage.fromarray(wrist_depth_img).resize((256, 256), PILImage.Resampling.BICUBIC)


        # already contains the gripper position
        state = torch.tensor(joint_positions, device=torch.device('cuda:0')).unsqueeze(0).float()

        # format to torch tensors
        base_rgb_img = torch.tensor(base_rgb_img, device=torch.device('cuda:0')).permute(2, 0, 1).unsqueeze(0).float() / 255
        base_depth_img = torch.tensor(base_depth_img, device=torch.device('cuda:0')).permute(2, 0, 1).unsqueeze(0).float() / 255
        wrist_rgb_img = torch.tensor(wrist_rgb_img, device=torch.device('cuda:0')).permute(2, 0, 1).unsqueeze(0).float() / 255
        wrist_depth_img = torch.tensor(wrist_depth_img, device=torch.device('cuda:0')).permute(2, 0, 1).unsqueeze(0).float() / 255

        # Resize images
        base_rgb_img = self.image_resizer.forward(base_rgb_img)
        base_depth_img = self.image_resizer.forward(base_depth_img)
        wrist_rgb_img = self.image_resizer.forward(wrist_rgb_img)
        wrist_depth_img = self.image_resizer.forward(wrist_depth_img)

        formatted_obs = {
            "observation.images.base.rgb": base_rgb_img,
            "observation.images.wrist.rgb": wrist_rgb_img,
            "observation.images.base.depth": base_depth_img,
            "observation.images.wrist.depth": wrist_depth_img,
            "observation.state": state
        }
        action = self.policy.select_action(formatted_obs)

        action = action.squeeze().detach().cpu().numpy()

        # append gripper position dummy if the gripper was not controlled by the policy (output dim = 6)
        if len(action) == 6:
            action = np.append(action, 0.0)
        
        return action
    
class LeRobotReplayAgent(Agent):
    def __init__(self, dataset_name: str, episode_idx: int) -> None:
        self.dataset = LeRobotDataset(repo_id=None, root=f"/home/zico/.cache/huggingface/lerobot/{dataset_name}", episodes=[episode_idx])
        self.actions = self.dataset.hf_dataset.select_columns("action")
        self.current_episode_idx = 0
        print(f"Replaying episode {episode_idx} with {self.dataset.num_frames} frames")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        takes an action from the lerobot dataset and returns it in gello format to the robot (this happens indefinately)
        """
        if self.current_episode_idx >= self.dataset.num_frames:
            self.current_episode_idx = 0

        action = self.actions[self.current_episode_idx]["action"].numpy()

        self.current_episode_idx += 1
        
        return action
    


class LeRobotTactileAgent(Agent):
    def __init__(self, policy) -> None:
        self.policy = policy

    def act(self, obs: dict) -> np.ndarray:
        """
        takes in a gello observation, turns it into a Lerobot observation and returns the action in gello format
        """

        base_image = obs["base_rgb"]
        wrist_image = obs["wrist_rgb"]
        #wrist_image = np.zeros((480, 640, 3))# obs["wrist_rgb"]
        joint_positions = obs["joint_positions"]
        fingertips = obs["fingertips"]
        accelerometer = obs["accelerometer"]

        # fingertips = np.random.randint(25,35,(32,))
        # fingertips = [16, 35, 40, 34, 31, 15, 39, 38, 44, 8, 22, 35, 49, 38, 11, 26, 33, 28, 6, 24, 27, 28, 25, 30, 32, 26, 26, 27, 31, 28, 18, 24]
        # accelerometer = 0


        # already contains the gripper position
        state = np.concatenate([joint_positions, fingertips, [accelerometer]], axis=0)
        state = torch.tensor(state).unsqueeze(0).float()

        # format to torch tensors
        base_image = torch.tensor(base_image).permute(2, 0, 1).unsqueeze(0).float() / 255
        wrist_image = torch.tensor(wrist_image).permute(2, 0, 1).unsqueeze(0).float() / 255

        formatted_obs = {
            "observation.images.base.rgb": base_image,
            "observation.images.wrist.rgb": wrist_image,
            "observation.state": state
        }
        action = self.policy.select_action(formatted_obs)

        action = action.squeeze().detach().numpy()

        # append gripper position dummy if the gripper was not controlled by the policy (output dim = 6)
        if len(action) == 6:
            action = np.append(action, 0.0)
        
        return action

    
    
   
    
if __name__ == "__main__":
    checkpoint_path = "/home/tlips/Code/gello_software/lerobot-output/checkpoints/gello-planar-push-last"
    policy = load_act_policy(checkpoint_path)

    observation = {
        "observation.images.base.rgb" : torch.rand(1,3, 480, 640),
        "observation.images.wrist.rgb" : torch.rand(1,3, 480, 640),
        "observation.state" : torch.rand(1, 7),
    }

    action = policy.select_action(observation)
    print(action)