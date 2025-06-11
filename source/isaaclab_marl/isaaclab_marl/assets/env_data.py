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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab_marl.config import APPLY_RANDOMIZATION, FIELD_LEVEL_SETTINGS, GOAL_LEVEL_SETTINGS, KICK_MODEL

if TYPE_CHECKING:
    from torch import Tensor


@configclass
class SoccerBallCfg:
    ownership_radius = 0.25
    reset_type = "no_speed"  # "no_speed" or "keep_speed"
    randomize_friction = True

    if APPLY_RANDOMIZATION:
        roll_friction_range = (0.00006, 0.0001)
        friction_range = (0.0, 0.5)
    else:
        roll_friction_range = (0.00005, 0.00005)
        friction_range = (0.25, 0.25)

    if KICK_MODEL == "default":
        kick_manipulation_type = "force"  # "force" or "velocity"
    else:
        kick_manipulation_type = "velocity"

    kick_force_gain = 0.5
    kick_area = "default"  # default, fullcircle
    kick_margin = 0.2


@configclass
class SoccerAgentsCfg:
    if APPLY_RANDOMIZATION:
        friction_range = (0.0, 0.2)
    else:
        friction_range = (0.1, 0.1)

    translation_shape = "ellipse"  # "ellipse" or "square"

    position_control_pid_gain = [20.0, 0.2]
    velocity_control_pid_gain = [20.0, 0.2]
    velocity_p_gain = 3
    translation_force_d_gain = 10
    rotation_torque_d_gain = 0.2

    cmd_speed_limit = [
        [0.35, -0.25],
        [0.25, -0.25],
        [2.5, -2.5],
    ]  # action limit 0, 1, 2


@configclass
class SoccerGameCfg:
    num_teams: int = 2  # 2 teams
    num_agents_per_team: int = 3  # max 5 players per team
    num_actions = 5

    # 0 for no control, 1 for policy, 2 for policy replay, 3 for bots
    default_control = [1, 2]  # blue, red agents
    agent_id_control = None  # [[1, 1 ... ],[2, 2 ...]] default if not set

    eval_env_ratio = 0.15

    num_agents_range = [1, num_agents_per_team]

    level_settings = [[2, 2], [3, 2], [3, 3], [3, 4]]

    ball = SoccerBallCfg()
    agents = SoccerAgentsCfg()

    create_field_visuals = False

    def __post_init__(self):
        if not self.agent_id_control:
            self.agent_id_control = [
                [control for _ in range(self.num_agents_per_team)] for control in self.default_control
            ]


class EnvCoreData:
    def __init__(self, cfg: SoccerGameCfg, num_envs, device):
        self.cfg = cfg
        self.num_envs = num_envs  # Denote als N
        self.num_teams = cfg.num_teams  # Always 2
        self.num_agents_per_team = cfg.num_agents_per_team  # Denote als N_T
        self.num_actions = cfg.num_actions

        self.num_agents_per_env = self.num_teams * self.num_agents_per_team
        self.num_agents = self.num_envs * self.num_agents_per_env

        self.device = device

        # common base shapes MARL data tensors.
        self.env_base_shape = (self.num_envs,)
        self.team_base_shape = (self.num_envs, self.num_teams)
        self.agent_base_shape = self.team_base_shape + (self.num_agents_per_team,)
        self.agent2agent_base_shape = self.agent_base_shape + (self.num_agents_per_team,)

        self.all_agent2agent_base_shape = (
            self.num_envs,
            self.num_agents_per_env,
            self.num_agents_per_env,
        )

        self.agent_pos_shape = self.agent_base_shape + (2,)
        self.agent_pose_shape = self.agent_base_shape + (3,)
        self.agent_quat_shape = self.agent_base_shape + (4,)

        self.env_flatten_base_shape = (self.num_envs * self.num_agents_per_env,)
        self.team_flatten_base_shape = (self.num_envs, self.num_agents_per_env)

        self.agent2agent_pose_shape = self.agent2agent_base_shape + (3,)

        self.env_flatten_pos_shape = self.env_flatten_base_shape + (2,)
        self.env_flatten_pose_shape = self.env_flatten_base_shape + (3,)
        self.env_flatten_quat_shape = self.env_flatten_base_shape + (4,)

        self.team_flatten_pos_shape = self.team_flatten_base_shape + (2,)
        self.team_flatten_pose_shape = self.team_flatten_base_shape + (3,)
        self.team_flatten_quat_shape = self.team_flatten_base_shape + (4,)

    def expand_to_env(self, tensor: Tensor, flatten_type: str = None):
        def expand(tensor: Tensor):
            if tensor.shape == self.env_base_shape:
                return (
                    tensor.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(
                        self.env_base_shape
                        + (
                            2,
                            self.num_agents_per_team,
                        )
                    )
                )
            elif tensor.shape == self.team_base_shape:
                return tensor.unsqueeze(-1).expand(self.team_base_shape + (self.num_agents_per_team,))
            else:
                raise Exception(tensor.shape + " cannot be expanded or flattened into the desired shape ")

        if flatten_type == "env":
            return expand(tensor).flatten()
        elif flatten_type == "team":
            return expand(tensor).reshape(self.team_flatten_base_shape)
        elif flatten_type == None:
            return expand(tensor)

    def expand_vector_to_env(self, tensor: Tensor, flatten_type: str = None):
        def expand_vector(tensor: Tensor):
            if tensor.shape[:-1] == self.env_base_shape:
                return (
                    tensor.unsqueeze(-2)
                    .unsqueeze(-2)
                    .expand(
                        self.env_base_shape
                        + (
                            2,
                            self.num_agents_per_team,
                            tensor.shape[-1],
                        )
                    )
                )
            elif tensor.shape[:-1] == self.team_base_shape:
                return tensor.unsqueeze(-2).expand(
                    self.team_base_shape
                    + (
                        self.num_agents_per_team,
                        tensor.shape[-1],
                    )
                )
            else:
                raise Exception(tensor.shape + " cannot be expanded or flattened into the desired shape ")

        if flatten_type == "env":
            return expand_vector(tensor).reshape(self.env_flatten_base_shape + (tensor.shape[-1],))
        elif flatten_type == "team":
            return expand_vector(tensor).reshape(self.team_flatten_base_shape + (tensor.shape[-1],))
        elif flatten_type == None:
            return expand_vector(tensor)

    def combine_index(self, env_ids=None, flag=None, return_index=True):
        if env_ids is None:
            env_ids = flag
        else:
            env_ids_expand = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            env_ids_expand[env_ids] = True
            env_ids = torch.logical_and(flag, env_ids_expand)
        if return_index:
            return env_ids.clone().nonzero(as_tuple=False).flatten()
        else:
            return env_ids.clone()


class EnvData(EnvCoreData):
    field_curriculum_level: Tensor = None
    env_origins: Tensor = None

    def __init__(self, cfg: SoccerGameCfg, num_envs, device):
        super().__init__(cfg, num_envs, device)

        self.num_eval_envs = int(self.num_envs * self.cfg.eval_env_ratio)
        self.team_size_curriculum_level = 0
        self.team_size_range = [
            [
                [self.cfg.num_agents_range[0], self.cfg.num_agents_range[1]],
                [self.cfg.num_agents_range[0], self.cfg.num_agents_range[1]],
            ]
        ]

        self.agent_ids_control = torch.tensor(self.cfg.agent_id_control, device=self.device, dtype=torch.int32)
        self.num_training_agents_per_env = torch.sum(self.agent_ids_control.flatten() == 1).item()
        self.num_policy_replay_agents_per_env = torch.sum(self.agent_ids_control.flatten() == 2).item()
        self.num_bots_per_env = torch.sum(self.agent_ids_control.flatten() == 3).item()

        self.field_width = 3.6
        self.init_tensors()

    def init_tensors(self):
        # Env base shaped tensors (N,)
        self.is_training_env = torch.zeros(self.env_base_shape, device=self.device, dtype=bool)

        self.goal_status = torch.zeros_like(self.is_training_env, dtype=torch.int32)

        self.field_length = torch.zeros_like(self.is_training_env, dtype=torch.float32)
        self.goal_width = torch.zeros_like(self.is_training_env, dtype=torch.float32)

        # (N, 2,)
        self.goal_post_pos = torch.zeros(self.env_base_shape + (2,), device=self.device, dtype=torch.float32)

        # Team base shaped tensors (N, 2)
        self.score = torch.zeros(self.team_base_shape, device=self.device, dtype=torch.int32)
        self.accumulated_score = torch.zeros_like(self.score)
        self.active_agent_per_team = torch.zeros_like(self.score)

        # Agent base shaped tensors (N, 2, N_T,)
        self.active = torch.zeros(self.agent_base_shape, device=self.device, dtype=torch.bool)
        self.inactive = torch.zeros_like(self.active)
        self.agent_control_type = torch.zeros_like(self.active, dtype=torch.int32)

        # Env flattened base shape (N * 2 * N_T,)
        self.training_actor_index = torch.zeros(self.env_flatten_base_shape, device=self.device, dtype=torch.int32)
        self.eval_actor_index = torch.zeros_like(self.training_actor_index)

        # The tensors are filled with initial value
        shuffled_env_index = torch.randperm(self.num_envs, device=self.device, dtype=torch.int32)
        if self.num_eval_envs == 0:
            self.is_training_env[:] = True
        else:
            self.is_training_env[shuffled_env_index[: -self.num_eval_envs]] = True
        self.is_eval_env = torch.logical_not(self.is_training_env)

        for i, control_type in enumerate(self.cfg.default_control):
            self.agent_control_type[:, i, :] = control_type

        self.regenerate_level_config(self.is_training_env)
        self.active_agent_per_team[self.is_eval_env] = torch.randint(
            low=self.cfg.num_agents_range[0],
            high=self.cfg.num_agents_range[1] + 1,
            size=(len(self.is_eval_env.nonzero()), 2),
            device=self.device,
            dtype=torch.int32,
        )
        self.regenerate_mask_index()

    def set_env_origin(self, env_origins):
        self.env_origins = env_origins

    def step(self, goal_status):
        self.score[goal_status == 1, 0] += 1
        self.score[goal_status == -1, 1] += 1

    def set_field_curriculum_level(self, field_curriculum_level):
        self.field_curriculum_level = field_curriculum_level

    def update_field(self, env_ids=None):
        def get_field_length(new_level):
            return FIELD_LEVEL_SETTINGS[self.cfg.level_settings[new_level][0]]

        def get_goal_width(new_level):
            return GOAL_LEVEL_SETTINGS[self.cfg.level_settings[new_level][1]]

        if env_ids is None:
            env_ids = ...

        self.field_length[env_ids] = torch.tensor(
            [get_field_length(i) for i in self.field_curriculum_level[env_ids]], device=self.device, dtype=torch.float32
        )
        self.goal_width[env_ids] = torch.tensor(
            [get_goal_width(i) for i in self.field_curriculum_level[env_ids]], device=self.device, dtype=torch.float32
        )
        self.goal_post_pos[env_ids, 0] = self.field_length[env_ids] / 2
        self.goal_post_pos[env_ids, 1] = self.goal_width[env_ids] / 2

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = ...

        self.update_field(env_ids)
        self.goal_status[env_ids] = 0
        self.accumulated_score[env_ids, :] += self.score[env_ids, :]
        self.score[env_ids, :] = 0

    def regenerate_level_config(self, env_ids: Tensor = None):
        num_regenerate_env = self.num_envs
        if env_ids is None:
            env_ids = ...
        else:
            num_regenerate_env = len(env_ids.nonzero())

        current_settings = self.team_size_range[min(self.team_size_curriculum_level, len(self.team_size_range) - 1)]
        for i in range(self.num_teams):
            self.active_agent_per_team[env_ids, i] = torch.randint(
                low=current_settings[i][0],
                high=current_settings[i][1] + 1,
                size=(num_regenerate_env,),
                device=self.device,
                dtype=torch.int32,
            )

    def regenerate_mask_index(self):
        for i in range(self.num_agents_per_team):
            self.active[..., i] = self.active_agent_per_team > i
        self.inactive = torch.logical_not(self.active)
        active_train = torch.logical_and(
            self.active.flatten(), self.expand_to_env(self.is_training_env, flatten_type="env")
        )
        active_eval = torch.logical_and(self.active.flatten(), self.expand_to_env(self.is_eval_env, flatten_type="env"))

        # Env flattened base shape (N * 2 * N_T,)
        self.training_actor_index = torch.logical_and(self.agent_control_type.flatten() == 1, active_train)
        self.eval_actor_index = torch.logical_and(self.agent_control_type.flatten() == 1, active_eval)
        self.policy_replay_actor_index = torch.logical_and(self.agent_control_type.flatten() == 2, active_train)
        self.bots_actor_index = torch.logical_or(
            torch.logical_and(self.agent_control_type.flatten() == 3, active_train),
            torch.logical_and(self.agent_control_type.flatten() != 1, active_eval),
        )
