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

import torch

from isaaclab_marl.utils.math_utils import mirror_agent_pose


class InitPosManager:
    def __init__(self, num_envs, num_agents_per_team, device):
        self.random_sampling = True
        mirror_init_pos = False
        self.init_rot_noise = torch.pi
        self.blue_default_pos_range = torch.tensor(
            [
                [[0.1, 0.95], [-0.5, 0.5], [-self.init_rot_noise, self.init_rot_noise]],
                [[-0.1, 0.5], [-0.95, 0.0], [-self.init_rot_noise, self.init_rot_noise]],
                [[-0.1, 0.5], [0.0, 0.95], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.3], [-0.95, -0.1], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.3], [0.1, 0.95], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [-0.95, -0.0], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [0.0, 0.95], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [-0.95, -0.0], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [0.0, 0.95], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [-0.95, -0.0], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [0.0, 0.95], [-self.init_rot_noise, self.init_rot_noise]],
            ],
            device=device,
        )
        self.red_default_pos_range = torch.tensor(
            [
                [[0.1, 0.95], [-0.5, 0.5], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.1, 0.5], [-0.95, 0.0], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.1, 0.5], [0.0, 0.95], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.3], [-0.95, -0.1], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.3], [0.1, 0.95], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [-0.65, -0.0], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [0.0, 0.65], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [-0.55, -0.0], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [0.0, 0.55], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [-0.55, -0.0], [-self.init_rot_noise, self.init_rot_noise]],
                [[0.0, 0.9], [0.0, 0.55], [-self.init_rot_noise, self.init_rot_noise]],
            ],
            device=device,
        )

        self.ball_default_pos_range = torch.tensor(
            [
                [[0.25, 0.75], [-0.75, 0.75]],
                [[0.15, 0.75], [-0.75, 0.75]],
                [[0.05, 0.75], [-0.75, 0.75]],
                [[-0.05, 0.75], [-0.75, 0.75]],
                [[-0.15, 0.75], [-0.75, 0.75]],
                [[-0.25, 0.75], [-0.75, 0.75]],
                [[-0.25, 0.65], [-0.75, 0.75]],
                [[-0.25, 0.55], [-0.75, 0.75]],
                [[-0.25, 0.45], [-0.75, 0.75]],
                [[-0.25, 0.25], [-0.75, 0.75]],
                [[-0.25, 0.35], [-0.75, 0.75]],
            ],
            device=device,
        )
        if mirror_init_pos:
            self.red_default_pos_range = self.blue_default_pos_range.clone()
            self.ball_default_pos_range[:] = 0

        self.current_level = 0
        self.max_level = self.ball_default_pos_range.shape[0] - 1
        self.num_agents_per_team = num_agents_per_team

        self.blue_default_pos = torch.zeros(num_envs, num_agents_per_team, 3, device=device)
        self.red_default_pos = torch.zeros_like(self.blue_default_pos)

        self.ball_default_pos = torch.zeros(num_envs, 2, device=device)
        self.re_sample_init_pos()

    def re_sample_init_pos(self, env_ids=None):
        if env_ids is None:
            env_ids = ...
        if self.random_sampling:
            self.blue_default_pos[env_ids, ...] = self.uniform_sample(
                self.blue_default_pos[env_ids, ...],
                self.blue_default_pos_range[: self.num_agents_per_team, :, 0],
                self.blue_default_pos_range[: self.num_agents_per_team, :, 1],
            )
            self.red_default_pos[env_ids, ...] = self.uniform_sample(
                self.red_default_pos[env_ids, ...],
                self.red_default_pos_range[: self.num_agents_per_team, :, 0],
                self.red_default_pos_range[: self.num_agents_per_team, :, 1],
            )
            self.red_default_pos[env_ids, ...] = mirror_agent_pose(self.red_default_pos[env_ids, ...])
            self.ball_default_pos[env_ids, :] = self.uniform_sample(
                self.ball_default_pos[env_ids, :],
                self.ball_default_pos_range[self.current_level, :, 0],
                self.ball_default_pos_range[self.current_level, :, 1],
            )
        else:
            self.blue_default_pos[env_ids, ...] = torch.mean(
                self.blue_default_pos_range[: self.num_agents_per_team], dim=-1
            )
            self.red_default_pos[env_ids, ...] = torch.mean(
                self.red_default_pos_range[: self.num_agents_per_team], dim=-1
            )
            self.red_default_pos[env_ids, ...] = mirror_agent_pose(self.red_default_pos[env_ids, ...])
            self.ball_default_pos[env_ids, :] = torch.mean(self.ball_default_pos_range[self.current_level], dim=-1)

    def uniform_sample(self, tensor, high, low):
        return torch.rand_like(tensor).mul(high - low).add(low)
