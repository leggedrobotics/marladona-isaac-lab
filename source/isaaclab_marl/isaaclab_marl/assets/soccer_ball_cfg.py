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
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_marl.config import MARL_ROOT_DIR


@configclass
class BallCfg(RigidObjectCfg):
    cls_name = "SoccerBall"
    asset_name = "ball"
    fix_base_link = False
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{MARL_ROOT_DIR}/assets/ball/ball.usd",
        activate_contact_sensors=False,
    )
    flip_visual_attachments = False

    init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3))
