# Imitation Learning for UR5e

The GELLO tele-operation device has been adopted to perform imitation learning on the UR5e at RoboHouse. The original repository has been forked and the following modifications has been made to accommodate our needs:

1. A `config.toml` has been added to easily configure the software for a particular mode of operation. In addition, quite a bit of software clean up was done for easier usage.
2. Support for a custom gripper was added, `Shadowtac` gripper (see `gello/robots/shadowtac_gripper.py`). In addition, support for operation without a gripper was also included.
3. LeRobot library was integrated. A dataset conversion script (see `scripts/gello_to_lerobot.py`) was added to convert from GELLO to LeRobot format. However, there is the option to record a dataset using the GELLO hardware directly in LeRobot format (see `dataset_type` in `config.toml`).
4. A LeRobot agent was added to allow for replay of episodes gathered, as well as running the policy once the model has been trained.

## Getting Started

## Dataset Generation

- Run `experiments/launch_camera_nodes.py` and `experiments/launch_nodes.py`, each in a separate process.
- Load `teleop` program on the UR5e.
- Run the program on the UR5e which will run the external control application which will connect to the host using RTDE interface.
- At this point the robot should move into its start pose.
- When you are ready to tele-operate, run the `experiments/run_env.py --agent gello` script in another process. Note that you will need to ensure the tele-operation device is roughly in the same pose as the robot. Also, if you want a re-run window to stream the data captured by the process, you can add `--display_data` argument.
- The terminal listens for key presses using `pynput` package. `s` starts a recording, `e` ends the recording, `d` discards the recording, and `q` ends the program.

Note that `experiments/quick_run.py` can be used to run all the processes needed at once. Also, keep in mind that `pynput` is not compatible with Wayland, so you may need to switch to Xorg if you are using a Wayland-based display server.

## Training

- Run the standard LeRobot training script for either `AcT`, `Diffusion`, or `SmolVLA`. An example: `lerobot-train --policy.push_to_hub=false --dataset.root=/home/<username>/.cache/huggingface/lerobot/ur5e_gello_cube_v2 --dataset.repo_id=lerobot/ur5e_gello_cube_v2 --policy.type=act --output_dir=./outputs/train/act_ur5e_gello_cube_v2 --job_name=act_ur5e_gello_cube_v2 --policy.device=cuda --wandb.enable=false --policy.repo_id=lerobot/ur5e_gello_cube_v2/act_policy`.

## Inference

- Running inference on the actual robot is very similar to the process of recording the dataset. The same sequence of actions should be followed, however, the following config parameters need to be set accordingly:
    - `agent = "policy`
    - `policy = <model_type>`
    - `checkpoint_path = <path_to_checkpoint>`
    - `task = <task_description>` This should be the same as the task description when generating the dataset. It is only important for the VLM based models, like `SmolVLA`.

## Lessons Learned

- The tele-operation is harder than it looks. It does take some skill.
- The environment setup is very important. This includes, the placement of cameras and how visually distinct the scene objects are from each other. We should be able to tele-operate using the sensor information only, otherwise, it might be difficult for the robot to learn the correct policy.
- Using depth data does not work at all. This needs to be investigate because it might be needed at a later stage.
- The current (shadowtac) gripper is not ideal for imitation learning.