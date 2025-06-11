# Copyright 2025 Zichong Li, ETH Zurich

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import statistics
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from typing import TYPE_CHECKING

import git

from isaaclab_marl.config import REPO_ROOT_DIR

if TYPE_CHECKING:
    from isaaclab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv


def store_code_state(logdir, repositories):
    for repository_file_path in repositories:
        repo = git.Repo(repository_file_path, search_parent_directories=True)
        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree

        branch_name = repo.commit()
        try:
            branch_name = repo.active_branch.name
        except TypeError as e:
            print(repo_name, ": ", e)
        content = (
            f"--- git status ---\n{repo.git.status()} \n\n\n--- git commit ---\n{repo.commit()}\n\n\n--- git branch"
            f" ---\n{branch_name}\n\n\n--- git diff ---\n{repo.git.diff(t)}\n"
        )
        with open(os.path.join(logdir, f"{repo_name}_git.diff"), "x", encoding="utf-8") as f:
            f.write(content)


class LogManager:
    def __init__(self, env, alg, log_directory, command_args, env_cfg, train_cfg, device):
        self.env: SoccerMARLEnv = env
        self.alg = alg

        self.log_dir = log_directory
        self.save_args_dict = command_args.__dict__

        self.device = device

        self.env_cfg = env_cfg
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.policy_replay_cfg = train_cfg["policy_replay"]

        self.log_steps = (
            self.env.max_episode_length
        )  # with 30s, dt = 0.02, decimal = 4. the max reasonalble value is 375

        self.git_status_repos = [REPO_ROOT_DIR]
        self.writer = None

    def init_config(self, total_num_training_actors):
        self.tot_timesteps = 0
        self.tot_time = 0
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.total_num_training_actors = total_num_training_actors
        self.set_eval_info()

    def get_world_state(self):
        world_state = self.env.observation_manager._obs_buffer["world_state"]["world_state"]
        return world_state[torch.logical_not(self.env.env_data.is_training_env)]

    def init_trajectory_buffer(self):
        eval_world_state = self.get_world_state()
        self.num_eval_env = eval_world_state.shape[0]

        self.last_world_state_eval = torch.zeros_like(eval_world_state).unsqueeze(0).repeat(self.log_steps, 1, 1)
        self.world_state_eval_buffer = deque(maxlen=self.log_steps)
        self.episode_length_buf = torch.zeros_like(
            self.env.episode_length_buf[torch.logical_not(self.env.env_data.is_training_env)]
        )

    def set_eval_info(self):
        self.eval_info = dict()
        self.eval_info["log_steps"] = self.log_steps
        self.eval_info["team_config"] = self.env.env_data.active_agent_per_team[
            torch.logical_not(self.env.env_data.is_training_env)
        ]
        self.eval_info["score"] = self.env.env_data.score[torch.logical_not(self.env.env_data.is_training_env)]
        self.eval_info["episode_len"] = self.env.episode_length_buf[
            torch.logical_not(self.env.env_data.is_training_env)
        ]

    def update_trajectory_buffer(self, update_index, score):
        # for reduced_env_id in range(len(update_index.nonzero())):
        if len(update_index.nonzero() > 0) and len(self.world_state_eval_buffer) == self.log_steps:
            self.last_world_state_eval[:, update_index, :] = torch.stack(list(self.world_state_eval_buffer))[
                :, update_index, :
            ]
            self.eval_info["score"][update_index] = score[update_index]
            self.eval_info["episode_len"][update_index] = self.episode_length_buf[update_index]
        self.episode_length_buf[:] = self.env.episode_length_buf[torch.logical_not(self.env.env_data.is_training_env)]
        obs = self.get_world_state()
        self.world_state_eval_buffer.extend([obs])

    def save_eval_trajectory(self, it, label=None):
        eval_traj_folder = os.path.join(self.log_dir, "eval_traj")
        if not os.path.exists(eval_traj_folder):
            os.makedirs(eval_traj_folder)
        path = os.path.join(eval_traj_folder, f"eval_traj_{it}_{label}.pt")
        saved_dict = {
            "world_state_traj": self.last_world_state_eval,
            "iter": it,
            "infos": self.eval_info,
        }
        if len(self.world_state_eval_buffer) == self.log_steps:
            print("Saving eval trajectory")
            torch.save(saved_dict, path)
        if label == "curriculum":
            self.world_state_eval_buffer.clear()

    def init_writer(self):
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

    def init_log_folder(self):
        store_code_state(self.log_dir, self.git_status_repos)

    def save_model(self, it, infos=None, path=None):
        if path is None:
            path = os.path.join(self.log_dir, f"model_{it}.pt")
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": it,
            "infos": infos,
        }

        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, it)

    def save_curriculum_state(self, it, level):
        curriculum_level_folder = os.path.join(self.log_dir, "curriculum_levels")
        if not os.path.exists(curriculum_level_folder):
            os.makedirs(curriculum_level_folder)
        path = os.path.join(curriculum_level_folder, f"model_{it}_level{level}.pt")
        self.save_model(it, path=path)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.total_num_training_actors
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f"{'Mean episode rewards:':>{pad}}"
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f""" {key}: {value:.4f} """
            ep_string += "\n"
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            * self.env.env_data.num_agents_per_env
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/entropy", locs["entropy_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["scorebufferBlue"]) > 0:
            self.writer.add_scalar("Train/mean_score_blue_all", statistics.mean(locs["scorebufferBlue"]), locs["it"])
            self.writer.add_scalar("Train/mean_score_red_all", statistics.mean(locs["scorebufferRed"]), locs["it"])
            self.writer.add_scalar("Eval/mean_score_blue_all", statistics.mean(locs["scorebufferBlueEval"]), locs["it"])
            self.writer.add_scalar("Eval/mean_score_red_all", statistics.mean(locs["scorebufferRedEval"]), locs["it"])

        reward_buffer = locs["metric_buffer_dict"]["reward"]
        reward_buffer_eval = locs["metric_buffer_dict_eval"]["reward"]
        self.writer.add_scalar("Train/init_pos_level", locs["init_pos_level"], locs["it"])

        for name in locs["metric_name_list"]:
            if len(reward_buffer) > 0:
                self.writer.add_scalar(
                    f"Train/mean_{name}", statistics.mean(locs["metric_buffer_dict"][name]), locs["it"]
                )
            if len(reward_buffer_eval) > 0:
                self.writer.add_scalar(
                    f"Eval/mean_{name}", statistics.mean(locs["metric_buffer_dict_eval"][name]), locs["it"]
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s, total {self.tot_time:.2f}s, ETA: {self.tot_time / (locs['it'] + 1) * (
                            locs['num_learning_iterations'] - locs['it']):.1f}s)\n"""
        )

        log_string += ep_string
        print(log_string)
