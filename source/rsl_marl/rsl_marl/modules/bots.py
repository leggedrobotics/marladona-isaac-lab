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

from isaaclab_marl.utils.math_utils import rotate2d

if TYPE_CHECKING:
    from isaaclab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv


class Bots:
    def __init__(self, env: SoccerMARLEnv, device) -> None:
        self.env = env

        self.envs_info = env.unwrapped.env_data
        self.device = device

        self.circle_around_distance = 0.35  # m
        self.alignment_angle_offset = 1.5  # rad
        self.kick_trigger_distance = 0.15

    def act(self, obs, previous_action, env_ids, configs=["keeper", "player", "player"]):
        base_pose_w = obs["base_own_pose_w"][env_ids]
        ball_pos_b = obs["ball_pos_b"][env_ids]
        ball_pos_w_buffer = obs["ball_pos_w"][env_ids]
        closest_agent_to_ball = obs["closest_agent_to_ball"][env_ids]
        player_id = obs["player_id"][env_ids]
        field_infos = obs["field_infos"][env_ids]

        player_previous_action = previous_action[env_ids].clone()

        for player_idx, cfg in enumerate(configs):
            selected_actor = player_id == player_idx
            selected_base_pose_w = base_pose_w[selected_actor]
            selected_ball_pos_b = ball_pos_b[selected_actor]
            selected_ball_pos_w_buffer = ball_pos_w_buffer[selected_actor]
            selected_closest_idx = closest_agent_to_ball[selected_actor]
            selected_field_info = field_infos[selected_actor]

            if cfg == "player":
                player_previous_action[selected_actor] = self.player_apply_action(
                    selected_ball_pos_b,
                    selected_ball_pos_w_buffer,
                    selected_base_pose_w,
                    selected_field_info,
                    selected_closest_idx,
                    player_previous_action[selected_actor],
                )
            elif cfg == "keeper":
                player_previous_action[selected_actor] = self.keeper_apply_action(
                    selected_ball_pos_b,
                    selected_ball_pos_w_buffer,
                    selected_base_pose_w,
                    selected_field_info,
                    selected_closest_idx,
                    player_previous_action[selected_actor],
                )
        player_previous_action = torch.clip(player_previous_action, -1.0, 1.0)
        return player_previous_action

    def player_apply_action(self, ball_pos_b, ball_pos_w_buffer, own_pose_w, field_info, closest_idx, previous_action):
        ball_pos_w = ball_pos_w_buffer.view(-1, 2, 2)[..., -1]
        ball_pos_w_vel = ball_pos_w - ball_pos_w_buffer.view(-1, 2, 2)[..., 0]

        direction = torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])
        previous_action[:, 2] = direction * 10
        goal_pos_w = field_info[:, 3].unsqueeze(-1).repeat(1, 2)
        goal_pos_w[:, 1] = 0.0
        ball2goal_direction = normalize(goal_pos_w - ball_pos_w)
        agent2ball_direction = normalize(ball_pos_w - own_pose_w[:, :2])
        y = (
            ball2goal_direction[:, 1] * agent2ball_direction[:, 0]
            - ball2goal_direction[:, 0] * agent2ball_direction[:, 1]
        )
        x = (
            ball2goal_direction[:, 0] * agent2ball_direction[:, 0]
            + ball2goal_direction[:, 1] * agent2ball_direction[:, 1]
        )
        alignment_angle = torch.atan2(y, x)

        engage_ball_index = torch.logical_and(ball_pos_b[:, 0] > self.circle_around_distance, closest_idx)
        defend_index = torch.logical_not(closest_idx)
        target_w = ball_pos_w[engage_ball_index] * 1.0
        target_w[:, 1] += 25 * ball_pos_w_vel[engage_ball_index, 1]
        previous_action[engage_ball_index, :2] = self.goto_target(own_pose_w[engage_ball_index], target_w)
        previous_action[defend_index, :2] = self.goto_target(
            own_pose_w[defend_index],
            (ball_pos_w[defend_index] - goal_pos_w[defend_index]) / 2,
        )
        circle_around_index = torch.logical_and(
            torch.logical_and(
                ball_pos_b[:, 0] <= self.circle_around_distance,
                torch.abs(alignment_angle) > self.alignment_angle_offset,
            ),
            closest_idx,
        )
        previous_action[:, 0][circle_around_index] = (ball_pos_b[:, 0][circle_around_index] - 0.25) * 10

        circle_alignment_index = torch.logical_and(ball_pos_b[:, 0] <= self.circle_around_distance, closest_idx)
        previous_action[:, 1][circle_alignment_index] = -alignment_angle[circle_alignment_index] * 3

        aligned_index = torch.logical_and(
            ball_pos_b[:, 0] <= self.circle_around_distance,
            torch.abs(alignment_angle) < self.alignment_angle_offset,
        )
        previous_action[:, 0][aligned_index] = 1.0

        previous_action[:, 3:5] = rotate2d(ball2goal_direction[:, 0], ball2goal_direction[:, 1], own_pose_w[:, 2])

        return previous_action

    def keeper_apply_action(self, ball_pos_b, ball_pos_w_buffer, own_pose_w, field_info, closest_idx, previous_action):
        ball_pos_w = ball_pos_w_buffer.view(-1, 2, 2)[..., -1]
        ball_pos_w_vel = ball_pos_w - ball_pos_w_buffer.view(-1, 2, 2)[..., 0]
        field_length = field_info[:, 2]
        own_goal_pos_w = field_length.unsqueeze(-1).repeat(1, 2)
        own_goal_pos_w[:, 1] = 0.0
        ball2goal_direction = normalize(own_goal_pos_w - ball_pos_w)

        direction = torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])
        ball_approach_goal_index = torch.logical_and(
            ball_pos_w_vel[:, 0] > 0.0, (own_goal_pos_w - ball_pos_w).norm() < 1
        )
        intersection_point = ball_pos_w[:, 1]
        intersection_point[ball_approach_goal_index] += (
            (field_length[ball_approach_goal_index] - ball_pos_w[ball_approach_goal_index, 0])
            / ball_pos_w_vel[ball_approach_goal_index, 0]
            * ball_pos_w_vel[ball_approach_goal_index, 1]
        )
        ratio = (own_goal_pos_w - ball_pos_w).norm(dim=-1).clip(0.3, 2) / 2
        y_diff = torch.clip(
            intersection_point / 2 * ratio + intersection_point * (1 - ratio),
            field_info[:, 1],
            field_info[:, 0],
        )
        target_w = previous_action[:, :2] * 0.0
        target_w[:, 1] = y_diff
        target_w[:, 0] = (field_length - 0.2) * ratio + ball_pos_w[:, 0] * (1 - ratio)
        previous_action[:, 2] = direction * 3

        previous_action[:, 3][ball_pos_b[:, 0] < self.kick_trigger_distance] = 1.0
        previous_action[:, 4] = 0
        previous_action[:, 3:5] = rotate2d(-ball2goal_direction[:, 0], -ball2goal_direction[:, 1], own_pose_w[:, 2])

        previous_action[:, :2] = self.goto_target(own_pose_w, target_w)
        previous_action[:, 0][
            torch.logical_and(ball_pos_w[:, 0] > 0.5, torch.logical_and(ball_pos_b[:, 0] < 0.5, closest_idx))
        ] = 1.0
        return previous_action

    def goto_target(self, current_pose, target_w):
        pos_diff = target_w - current_pose[:, :2]
        return rotate2d(pos_diff[:, 0], pos_diff[:, 1], current_pose[:, 2]) * 10
