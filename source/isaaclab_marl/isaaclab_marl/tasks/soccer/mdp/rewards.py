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

from torch.nn.functional import normalize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv


def score(env: SoccerMARLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    ego_score = env.env_data.expand_to_env(env.env_data.goal_status).clone()
    ego_score[:, 1] *= -1
    return ego_score.view(env.env_data.env_flatten_base_shape)


def ball_outside_field_penalty(env: SoccerMARLEnv):
    return env.env_data.expand_to_env(
        torch.logical_and(~env.ball.ball_inside_field, env.ball.goal_status == 0) * -1.0, flatten_type="env"
    )


def vel_2_ball(env: SoccerMARLEnv, mode: str):
    vel_2_ball_reward = torch.sum(
        normalize(env.agents.ball_pos_b) * env.agents.base_lin_vel_b.view(env.env_data.env_flatten_pos_shape), dim=-1
    )
    if mode == "default":
        return vel_2_ball_reward
    elif mode == "until_close":
        vel_2_ball_reward = vel_2_ball_reward.view(env.env_data.agent_base_shape)
        ball_team_ownership = torch.zeros_like(vel_2_ball_reward)
        ball_player_ownership = torch.zeros_like(vel_2_ball_reward)

        ball_team_ownership[torch.arange(env.num_envs).unsqueeze(-1), env.ball.owning_team.unsqueeze(-1)] = 1
        ball_player_ownership[
            torch.arange(env.num_envs).unsqueeze(-1),
            torch.arange(2).unsqueeze(0),
            env.ball.owning_player.unsqueeze(-1),
        ] = 1
        return (vel_2_ball_reward * torch.logical_not(ball_team_ownership * ball_player_ownership)).view(
            env.env_data.env_flatten_base_shape
        )
    else:
        NotImplementedError


def vel_ball_2_goal(env: SoccerMARLEnv):
    goal_pos = env.env_data.goal_post_pos.clone()
    goal_pos[:, 1] = 0

    goal_direction = torch.cat(
        [
            normalize(-goal_pos - env.ball.base_pos_w).unsqueeze(1),
            normalize(goal_pos - env.ball.base_pos_w).unsqueeze(1),
        ],
        dim=1,
    )
    vel_ball_2_goal_reward = torch.sum(
        goal_direction
        * torch.cat([env.ball.base_lin_vel_w.unsqueeze(1), -env.ball.base_lin_vel_w.unsqueeze(1)], dim=1),
        dim=-1,
    )
    return env.env_data.expand_to_env(vel_ball_2_goal_reward, flatten_type="env")


def ball_direction(env: SoccerMARLEnv, std):
    return torch.exp(-torch.square(torch.atan2(env.agents.ball_pos_b[:, 1], env.agents.ball_pos_b[:, 0]) / std)) * 0.05


def collision_penalty(env: SoccerMARLEnv, collision_radius_agent, collision_radius_goal_post):
    agent2agent_collision = (
        torch.min(
            env.agents.agent2agent_distance,
            dim=-1,
        )[0]
        < collision_radius_agent
    )

    agent2object_collision = torch.logical_or(
        torch.logical_or(
            torch.abs(env.agents.base_pose_w[..., 0]) > env.env_data.expand_to_env(env.env_data.field_length) / 2 + 0.3,
            torch.abs(env.agents.base_pose_w[..., 1]) > 3.0 + 0.3,
        ),
        (
            torch.abs(env.agents.base_pose_w[..., :2]) - env.env_data.expand_vector_to_env(env.env_data.goal_post_pos)
        ).norm(dim=-1)
        < collision_radius_goal_post,
    )
    return agent2agent_collision.flatten() * -1.0 + agent2object_collision.flatten() * -1.0
