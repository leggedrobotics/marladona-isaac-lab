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

from isaaclab_marl.assets.env_data import EnvCoreData, EnvData

from .env_data import EnvCoreData, EnvData

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from .env_data import SoccerGameCfg


class SoccerBall(EnvCoreData):
    asset: RigidObject = None

    def __init__(self, cfg: SoccerGameCfg, env_data: EnvData, num_envs, device, asset):
        super().__init__(cfg, num_envs=num_envs, device=device)
        self.ball_cfg = cfg.ball
        self.asset = asset
        self.env_data = env_data
        self.init_tensor()

    def init_tensor(self):
        self.base_pos_w = torch.zeros(self.env_base_shape + (2,), device=self.device, dtype=torch.float32)
        self.base_default_pos_w = torch.zeros_like(self.base_pos_w)

        # rolling friction randomization property
        num_buckets = 64
        rolling_friction_buckets = torch.empty(num_buckets, device=self.device).uniform_(
            self.ball_cfg.roll_friction_range[0], self.ball_cfg.roll_friction_range[1]
        )
        bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
        self.rolling_friction = rolling_friction_buckets[bucket_ids]

        self.base_lin_vel_w = self.asset.data.root_lin_vel_w[..., :2]

        self.closest_agent_index = torch.zeros(self.team_base_shape, device=self.device, dtype=torch.int32)
        # self.closest_agent_distance = torch.zeros(self.team_base_shape, device=self.device, dtype=torch.float32)
        self.owning_team = -torch.ones(self.env_base_shape, device=self.device, dtype=torch.int32)
        # self.owning_player: minus: red team, 0: no owner, plus: blue team. value implies the player index
        self.owning_player = torch.zeros(self.env_base_shape, device=self.device, dtype=torch.int32)

    def step(self):
        self.base_pos_w = self.asset.data.root_pos_w[:, :2] - self.env_data.env_origins[:, :2]

        self.base_lin_vel_w = self.asset.data.root_lin_vel_w[..., :2]

        self.check_ball_location()

    def compute_ball2agent_states(self, agent_pose_w):
        ball2agent_distance = torch.cdist(
            self.base_pos_w.unsqueeze(1),
            agent_pose_w[..., :2].view(self.team_flatten_pos_shape),
            p=2,
        ).view(self.team_base_shape + (self.num_agents_per_team,))

        self.closest_agent_index = torch.argsort(ball2agent_distance, dim=-1)[..., 0]
        closest_agent_distance = ball2agent_distance[
            torch.arange(self.num_envs).unsqueeze(-1),
            torch.arange(2).unsqueeze(0),
            self.closest_agent_index.unsqueeze(1)[..., 0],
        ]

        self.owning_team[...] = -1
        self.owning_team[
            torch.logical_and(
                closest_agent_distance[:, 0] < self.ball_cfg.ownership_radius,
                closest_agent_distance[:, 1] > self.ball_cfg.ownership_radius,
            )
        ] = 0
        self.owning_team[
            torch.logical_and(
                closest_agent_distance[:, 1] < self.ball_cfg.ownership_radius,
                closest_agent_distance[:, 0] > self.ball_cfg.ownership_radius,
            )
        ] = 1
        closest_team_id = torch.argmin(closest_agent_distance, dim=1)
        self.owning_player[:] = self.closest_agent_index[
            torch.arange(self.num_envs),
            closest_team_id,
        ]

    def set_ball_default_pos(self, ball_default_pos):
        self.base_default_pos_w = ball_default_pos

    def reset(self, env_ids):
        root_states = self.asset.data.default_root_state[env_ids].clone()

        positions = self.env_data.env_origins[env_ids].clone()
        positions[:, 0] += self.base_default_pos_w[env_ids, 0] * self.env_data.field_length[env_ids] / 2
        positions[:, 1] += self.base_default_pos_w[env_ids, 1] * self.env_data.field_width / 2
        positions[:, 2] = self.asset.data.default_root_state[env_ids, 2]
        root_states[:, :3] = positions
        root_states[:, 7:] = 0.0
        self.asset.write_root_state_to_sim(root_states, env_ids=env_ids)

    def check_ball_location(self):
        self.ball_inside_field = torch.logical_and(
            torch.logical_and(
                self.base_pos_w[:, 0] < self.env_data.field_length / 2,
                self.base_pos_w[:, 0] > -self.env_data.field_length / 2,
            ),
            torch.logical_and(
                self.base_pos_w[:, 1] < self.env_data.field_width / 2,
                self.base_pos_w[:, 1] > -self.env_data.field_width / 2,
            ),
        )
        self.goal_status = torch.logical_and(
            self.base_pos_w[:, 1] < self.env_data.goal_width / 2,
            self.base_pos_w[:, 1] > -self.env_data.goal_width / 2,
        ) * (
            (self.base_pos_w[:, 0] > self.env_data.field_length / 2) * -1
            + (self.base_pos_w[:, 0] < -self.env_data.field_length / 2) * 1
        )

        self.ball_reset = torch.logical_and(
            torch.logical_not(self.ball_inside_field),
            self.goal_status == 0,
        )
        self.reset_ball_at_border(self.ball_reset.nonzero(as_tuple=False).squeeze(-1))

    def get_goal_status(self):
        return self.goal_status

    def reset_ball_at_border(self, env_ids):
        if len(env_ids) == 0:
            return
        ball_reset_tolerance = 0.005
        root_state = self.asset.data.root_state_w[env_ids].clone()
        # self.get_default_root_state(env_ids)
        root_state[:, :2] = self.env_data.env_origins[env_ids, :2].clone()
        violate_x_border_index = (
            torch.abs(self.base_pos_w[env_ids, 0]) > self.env_data.field_length[env_ids] / 2 - ball_reset_tolerance
        )
        violate_y_border_index = (
            torch.abs(self.base_pos_w[env_ids, 1]) > self.env_data.field_width / 2 - ball_reset_tolerance
        )
        last_base_pos_w = self.base_pos_w[env_ids]

        root_state[:, 0] += torch.clip(
            last_base_pos_w[:, 0],
            -self.env_data.field_length[env_ids] / 2 + ball_reset_tolerance,
            self.env_data.field_length[env_ids] / 2 - ball_reset_tolerance,
        )
        root_state[:, 1] += torch.clip(
            last_base_pos_w[:, 1],
            -self.env_data.field_width / 2 + ball_reset_tolerance,
            self.env_data.field_width / 2 - ball_reset_tolerance,
        )

        if self.cfg.ball.reset_type == "no_speed":
            root_state[:, 7:] = 0.0
            root_state[:, 2] = 0.3
        else:
            root_state[violate_x_border_index, 7] *= -1
            root_state[violate_y_border_index, 8] *= -1

            root_state[violate_y_border_index, 10] = 0.0
            root_state[violate_x_border_index, 11] = 0.0

        self.asset.write_root_state_to_sim(root_state, env_ids=env_ids)
