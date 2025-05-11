"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import numpy as np
import torch

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{normal_repr(self)} \n {self.shape}, {self.min()}, {self.max()}"
np.set_printoptions(edgeitems=3, linewidth=1000, threshold=100)
torch.set_printoptions(edgeitems=3, linewidth=1000, threshold=100)


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from rsl_marl.runners import OnPolicyRunner
from rsl_marl.utils.custom_vecenv_wrapper import CustomVecEnvWrapper

# Import extensions to set up environment tasks
import isaaclab_marl.tasks  # noqa: F401

from isaaclab_marl.config import WORKSPACE_ROOT_DIR

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_marl.tasks.soccer.soccer_marl_env_cfg import SoccerMARLEnvCfg
    from isaaclab_marl.tasks.soccer.agents.soccer_marl_ppo_runner_cfg import SoccerMARLPPORunnerCfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: SoccerMARLEnvCfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg: SoccerMARLPPORunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    agent_cfg.policy_replay.dynamic_generate_replay_level = False

    # specify directory for logging experiments
    log_root_path = os.path.join(WORKSPACE_ROOT_DIR, "../example_policy")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = CustomVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, command_args=args_cli)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs_dict = env.get_observations()
    timestep = 0
    env_data = env.unwrapped.env_data
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            obs = torch.cat([obs_dict["policy"], obs_dict["neighbor"]], dim=1)

            actions = policy(obs)

            obs_dict, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
