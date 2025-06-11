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

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab_marl.config import FIELD_LEVEL_SETTINGS, GOAL_LEVEL_SETTINGS, SCALING

if TYPE_CHECKING:
    from .soccer_terrain_cfg import SubTerrainsCfg, TerrainCfg


def border_mesh(
    outer_length: float, outer_width: float, inner_length: float, inner_width: float, height: float, pos=[0.0, 0.0, 0.0]
):
    """
    Creates a rectangular border mesh with a rectangular hole

         ________________________
        |                        |
        |      ____________      | w
        |     |            |     | i
        |     |            |     | d
        |     |____________|     | t
        |                        | h
        |________________________|
                 length


    """
    thickness_x = (outer_length - inner_length) / 2
    tickness_y = (outer_width - inner_width) / 2

    dims = [outer_length, tickness_y, height]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0], pos[1] + (tickness_y + inner_width) / 2, pos[2]]
    box_north = trimesh.creation.box(dims, pose)
    pose[:3, -1] = [pos[0], pos[1] - (tickness_y + inner_width) / 2, pos[2]]
    box_south = trimesh.creation.box(dims, pose)
    dims = [thickness_x, outer_width, height]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0] + (thickness_x + inner_length) / 2, pos[1], pos[2]]
    box_east = trimesh.creation.box(dims, pose)
    pose[:3, -1] = [pos[0] - (thickness_x + inner_length) / 2, pos[1], pos[2]]
    box_west = trimesh.creation.box(dims, pose)
    boxes = [box_east, box_west, box_north, box_south]
    return boxes


def cylinder_mesh(pos, radius=0.049, height=0.79):
    pose = np.eye(4)
    pos[2] = height / 2
    pose[0:3, -1] = np.array(pos)
    return trimesh.creation.cylinder(radius=radius, height=height, transform=pose)


def soccer_field(difficulty, cfg: "TerrainCfg"):
    border_size = 0.01
    difficulty = int(difficulty * len(cfg.level_settings))
    field_length = FIELD_LEVEL_SETTINGS[cfg.level_settings[difficulty][0]]
    field_width = 7.4 * SCALING
    goal_width = GOAL_LEVEL_SETTINGS[cfg.level_settings[difficulty][1]]
    border_lenght = field_length / 9 * 10.4
    pos = [0, 0, 0]
    meshes = []
    barrier_height = 0.25
    border_meshes = border_mesh(
        border_lenght,
        field_width,
        border_lenght - 2 * (border_size),
        field_width - 2 * (border_size),
        0.5,
        [pos[0], pos[1], barrier_height / 2],
    )
    meshes += border_meshes
    x = field_length / 2
    y = goal_width / 2
    goal_offsets = [[x, y, 0], [x, -y, 0], [-x, -y, 0], [-x, y, 0]]
    meshes += [cylinder_mesh(np.array(pos) + np.array(goal_offsets[i])) for i in range(4)]

    meshes = trimesh.util.concatenate(meshes)

    return meshes, pos


def plane(difficulty, cfg: "SubTerrainsCfg.PlaneCfg"):
    x0 = [cfg.length, cfg.width, 0.0]
    x1 = [cfg.length, 0.0, 0.0]
    x2 = [0.0, cfg.width, 0.0]
    x3 = [0.0, 0.0, 0.0]
    vertices = np.array([x0, x1, x2, x3])
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.apply_translation([-0.5 * cfg.length, -0.5 * cfg.width, 0.0])
    return mesh, [0.0, 0.0, 0.0]
