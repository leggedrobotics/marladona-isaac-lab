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

from isaaclab.managers import SceneEntityCfg

from isaaclab_marl.utils.math_utils import mirror_agent_pose

if TYPE_CHECKING:
    from isaaclab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv


# Helper functions
def expand_pose_rot(pose, expand_rot=False):
    rot = pose[..., 2:] % (2 * torch.pi)
    if expand_rot:
        return torch.cat([pose[..., :2], torch.sin(rot), torch.cos(rot)], dim=-1)
    else:
        return pose


def normalize_pose(pose, field_length, field_width, normalize=False):
    if normalize:
        pose[..., 0] = pose[..., 0] / field_length * 2
        pose[..., 1] = pose[..., 1] / field_width * 2
    return pose


# Observation Shape:
# (num_envs * num_agents_per_env, obs_dim,)
def base_own_vel(env: SoccerMARLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("agents")) -> torch.Tensor:
    ego_base_vel_w = torch.cat([env.agents.base_lin_vel_w, env.agents.base_ang_vel_w.unsqueeze(-1)], dim=-1)
    ego_base_vel_w[:, env.env_data.num_agents_per_team :] *= -1

    return ego_base_vel_w.view(env.env_data.env_flatten_pose_shape)


def ball_pos_w(
    env: SoccerMARLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"), normalize: bool = True
) -> torch.Tensor:
    pos = env.ball.base_pos_w.clone()
    if normalize:
        pos[:, 0] = pos[:, 0] / env.env_data.field_length * 2
        pos[:, 1] = pos[:, 1] / env.env_data.field_width * 2
    expanded_pos = env.env_data.expand_vector_to_env(pos).clone()
    expanded_pos[:, 1] *= -1
    return expanded_pos.reshape(env.env_data.env_flatten_pos_shape)


def ball_pos_b(env: SoccerMARLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")):
    return env.agents.ball_pos_b.view(env.env_data.env_flatten_pos_shape)


def ball_vel_w(env: SoccerMARLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ego_ball_vel_w = env.env_data.expand_vector_to_env(env.ball.base_lin_vel_w).clone()
    ego_ball_vel_w[:, 1] *= -1
    return ego_ball_vel_w.view(env.env_data.env_flatten_pos_shape)


def base_own_pose_w(
    env: SoccerMARLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("agents"),
    expand_rot: bool = False,
    normalize: bool = True,
):
    ego_base_pose = env.agents.base_pose_w.clone()
    ego_base_pose[:, 1] = mirror_agent_pose(ego_base_pose[:, 1])
    pose = expand_pose_rot(ego_base_pose, expand_rot)

    pose = pose.reshape(env.env_data.env_flatten_base_shape + tuple(pose.shape[-1:]))
    pose = normalize_pose(
        pose,
        env.env_data.expand_to_env(env.env_data.field_length, flatten_type="env"),
        env.env_data.field_width,
        normalize,
    )

    return pose


def field_infos(env: SoccerMARLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("agents"), normalize=True):
    field_infos = torch.zeros(env.env_data.env_base_shape + (5,), device=env.device, dtype=torch.float32)
    if normalize:
        field_infos[:, 0] = env.env_data.goal_width / env.env_data.field_width
        field_infos[:, 1] = -env.env_data.goal_width / env.env_data.field_width
        field_infos[:, 2] = 1
        field_infos[:, 3] = -1
        field_infos[:, 4] = 1
    else:
        field_infos[:, 0] = env.env_data.goal_width / 2
        field_infos[:, 1] = -env.env_data.goal_width / 2
        field_infos[:, 2] = env.env_data.field_length / 2
        field_infos[:, 3] = -env.env_data.field_length / 2
        field_infos[:, 4] = env.env_data.field_width / 2
    expanded_field_info = env.env_data.expand_vector_to_env(field_infos, flatten_type="env")

    active_agent_red_perspective = env.env_data.active_agent_per_team.clone()
    active_agent_red_perspective[:, 1] = env.env_data.active_agent_per_team[:, 0]
    active_agent_red_perspective[:, 0] = env.env_data.active_agent_per_team[:, 1]
    ego_active_agent = torch.cat(
        [env.env_data.active_agent_per_team.unsqueeze(1), active_agent_red_perspective.unsqueeze(1)], dim=1
    )
    expanded_ego_active_agent = env.env_data.expand_vector_to_env(ego_active_agent, flatten_type="env")
    return torch.cat([expanded_field_info, expanded_ego_active_agent], dim=-1).view(
        env.env_data.env_flatten_base_shape + (7,)
    )


def player_id(env: SoccerMARLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("agents")):
    player_id = torch.arange(env.env_data.num_agents_per_team, device=env.device)
    return player_id.repeat(env.num_envs, 2, 1).flatten()


def closest_teammate_pose(
    env: SoccerMARLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("agents"),
    expand_rot: bool = False,
    normalize: bool = True,
    max_num_neighbor: int = -1,
):
    closest_teammate_pose_w = env.agents.closest_teammate_pose_w[..., :-1, :]  # remove ego agent pose from the tensor
    if max_num_neighbor == -1:
        max_num_neighbor = closest_teammate_pose_w.shape[-2]
    closest_teammate_ego_pose = closest_teammate_pose_w[..., :max_num_neighbor, :].clone()
    closest_teammate_ego_pose[:, 1, ...] = mirror_agent_pose(closest_teammate_ego_pose[:, 1, ...])
    # num_teammate = closest_teammate_ego_pose.shape[-2]

    pose = expand_pose_rot(closest_teammate_ego_pose, expand_rot)
    pose = pose.reshape(env.env_data.env_flatten_base_shape + tuple(pose.shape[-2:]))
    pose = normalize_pose(
        pose,
        env.env_data.expand_to_env(env.env_data.field_length, flatten_type="env")
        .unsqueeze(-1)
        .expand(pose[..., 0].shape),
        env.env_data.field_width,
        normalize,
    )
    return pose.reshape(pose.shape[0], -1)


def closest_opponent_pose(
    env: SoccerMARLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("agents"),
    expand_rot: bool = False,
    normalize: bool = True,
    max_num_neighbor: int = -1,
):
    closest_opponent_ego_pose = env.agents.closest_opponent_pose_w[..., :max_num_neighbor, :].clone()
    closest_opponent_ego_pose[:, 1, ...] = mirror_agent_pose(closest_opponent_ego_pose[:, 1, ...])
    # num_teammate = closest_teammate_ego_pose.shape[-2]

    pose = expand_pose_rot(closest_opponent_ego_pose, expand_rot)
    pose = pose.reshape(env.env_data.env_flatten_base_shape + tuple(pose.shape[-2:]))
    pose = normalize_pose(
        pose,
        env.env_data.expand_to_env(env.env_data.field_length, flatten_type="env")
        .unsqueeze(-1)
        .expand(pose[..., 0].shape),
        env.env_data.field_width,
        normalize,
    )

    return pose.reshape(pose.shape[0], -1)


# Observation Shape:
# (num_envs, obs_dim,)
def world_state(env: SoccerMARLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("agents")):
    pose = env.agents.base_pose_w.view(env.env_data.num_envs, -1)
    ball_pos = env.ball.base_pos_w
    return torch.cat(
        [
            pose,
            ball_pos,
        ],
        dim=-1,
    )


def closest_agent_to_ball(
    env: SoccerMARLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("agents"),
):
    closest_agent_to_ball = torch.zeros(env.env_data.agent_base_shape, device=env.device, dtype=torch.int32)
    closest_agent_to_ball[
        torch.arange(env.num_envs).unsqueeze(-1), torch.arange(2).unsqueeze(0), env.ball.closest_agent_index
    ] = True
    return closest_agent_to_ball.flatten()
