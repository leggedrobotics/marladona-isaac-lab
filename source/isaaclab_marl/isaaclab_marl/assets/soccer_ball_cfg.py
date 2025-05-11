import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_marl.config import APPLY_RANDOMIZATION, KICK_MODEL, MARL_ROOT_DIR


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
