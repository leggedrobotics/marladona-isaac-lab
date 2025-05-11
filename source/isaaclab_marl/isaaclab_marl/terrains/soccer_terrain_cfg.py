from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_marl.config import FIELD_LEVEL_SETTINGS

from . import soccer_mesh_terrains as mt


@configclass
class SoccerFieldCfg(SubTerrainBaseCfg):
    function = mt.soccer_field
    level_settings = [[2, 2], [3, 2], [3, 3], [3, 4]]


@configclass
class TerrainCfg(TerrainGeneratorCfg):
    size = (10.4 * 0.8, 7.4 * 0.8)
    mesh_type: str = "trimesh"  # none, plane, heightfield or trimesh
    env_spacing = 3.0  # not used with heightfields/trimeshes
    border_size = 25  # [m]
    curriculum = True
    static_friction = 0.5
    dynamic_friction = 0.5
    restitution = 1.0
    max_init_terrain_level = 0  # starting curriculum state
    terrain_length = 10.4
    terrain_width = 7.4

    border_width = 0.0
    use_cache = False

    # Will be updated by the MultiAgentEnv at initialization
    num_rows = len(FIELD_LEVEL_SETTINGS)  # 8  # number of terrain rows (levels)
    default_eval_level = 2  # default evaluation level

    num_cols = 1  # number of terrain cols (types)
    # heightfield specific
    horizontal_scale = 0.1  # [m]
    vertical_scale = 0.005  # [m]
    slope_threshold = 0.75

    # slopes above this threshold will be corrected to vertical surfaces

    sub_terrains = {"soccer_field": SoccerFieldCfg()}
    create_field_visuals = False
