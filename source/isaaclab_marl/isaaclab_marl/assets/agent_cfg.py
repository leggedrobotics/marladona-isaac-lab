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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.utils import configclass

from isaaclab_marl.config import MARL_ROOT_DIR

from .from_files import spawn_from_usd


@configclass
class AgentCfg(RigidObjectCfg):
    cls_name = "Agent"
    asset_name = "agent"
    fix_base_link = False
    base_color = (1.0, 0.0, 0.0)
    spawn = sim_utils.UsdFileCfg(
        func=spawn_from_usd,
        usd_path=f"{MARL_ROOT_DIR}/assets/soccer_agent/soccer_agent.usd",
        activate_contact_sensors=False,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    flip_visual_attachments = False
    replace_cylinder_with_capsule = False
    collapse_fixed_joints = False

    init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1))

    def __post_init__(self):
        self.spawn.visual_material.diffuse_color = self.base_color


@configclass
class AgentTeamCfg(RigidObjectCollectionCfg):
    rigid_objects: dict[str, RigidObjectCfg] = dict()
