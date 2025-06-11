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

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz


@torch.jit.script
def rotate2d(x, y, r):
    x_new = torch.cos(r) * x + torch.sin(r) * y
    y_new = -torch.sin(r) * x + torch.cos(r) * y
    return torch.stack([x_new, y_new], dim=-1)


@torch.jit.script
def transform2dPos(ref_pose, pos):
    diff = pos - ref_pose[..., :2]
    return rotate2d(diff[..., 0], diff[..., 1], ref_pose[..., 2])


@torch.jit.script
def get_yaw_angle(quat):
    return euler_xyz_from_quat(quat)[2]


@torch.jit.script
def get_quat_from_yaw(yaw, euler_angle, env_ids=None):
    if env_ids is None:
        env_ids = ...
    euler_angle[0][env_ids] = 0.0
    euler_angle[1][env_ids] = 0.0
    return quat_from_euler_xyz(euler_angle[0][env_ids], euler_angle[1][env_ids], yaw)


@torch.jit.script
def mirror_agent_pose(base_pose):
    mirrored_base_pos = base_pose.clone()
    mirrored_base_pos[..., :2] = -1 * base_pose[..., :2]
    mirrored_base_pos[..., 2] = (base_pose[..., 2] + torch.pi) % (2 * torch.pi)
    return mirrored_base_pos
