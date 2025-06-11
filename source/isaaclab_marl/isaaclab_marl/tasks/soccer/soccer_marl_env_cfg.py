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

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Scene definition
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_marl.tasks.soccer.mdp as mdp
import isaaclab_marl.tasks.soccer.mdp.observations as O
import isaaclab_marl.tasks.soccer.mdp.rewards as R
import isaaclab_marl.tasks.soccer.mdp.terminations as T
from isaaclab_marl.assets.agent_cfg import AgentCfg, AgentTeamCfg
from isaaclab_marl.assets.env_data import SoccerGameCfg
from isaaclab_marl.assets.soccer_ball_cfg import BallCfg
from isaaclab_marl.config import FIELD_LEVEL_SETTINGS, GOAL_LEVEL_SETTINGS, MARL_ROOT_DIR, WORLD_POS_NORMALIZATION
from isaaclab_marl.managers.ma_action_manager import BallControlAction, BaseMovementAction

from isaaclab_marl.terrains import TerrainCfg  # isort: skip


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainCfg(),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    agents = AgentTeamCfg()
    ball: BallCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    def __post_init__(self):
        self.ground.spawn.size = (
            self.terrain.terrain_generator.size[0] * self.terrain.terrain_generator.num_rows + 5,
            self.terrain.terrain_generator.size[1] * self.terrain.terrain_generator.num_cols + 5,
        )
        self.ground.init_state.pos = (
            -self.terrain.terrain_generator.size[0] / 2,
            -self.terrain.terrain_generator.size[1] / 2,
            0.0,
        )

    def create_soccer_field_visual(self):
        level_settings = self.terrain.terrain_generator.sub_terrains["soccer_field"].level_settings
        for i in range(self.terrain.terrain_generator.num_rows):
            goal_scaler = GOAL_LEVEL_SETTINGS[level_settings[i][1]] / GOAL_LEVEL_SETTINGS[-1]
            field_scaler = FIELD_LEVEL_SETTINGS[level_settings[i][0]] / FIELD_LEVEL_SETTINGS[-1]
            for j in range(self.terrain.terrain_generator.num_cols):
                sub_terrain_origin_x = self.terrain.terrain_generator.size[0] * (
                    -self.terrain.terrain_generator.num_rows / 2 + i
                )
                sub_terrain_origin_y = self.terrain.terrain_generator.size[1] * (
                    -self.terrain.terrain_generator.num_cols / 2 + j
                )
                setattr(
                    self,
                    f"soccer_field_{i}_{j}",
                    AssetBaseCfg(
                        prim_path=f"/Visuals/SoccerFieldVisuals/soccer_field_{i}_{j}",
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(sub_terrain_origin_x, sub_terrain_origin_y, 0.0)),
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=MARL_ROOT_DIR + "/assets/field/field.usd",
                            scale=(0.6 * field_scaler, 0.6, 1.0),
                        ),
                    ),
                )

                goal_red_x = sub_terrain_origin_x - FIELD_LEVEL_SETTINGS[level_settings[i][0]] / 2
                goal_blue_x = sub_terrain_origin_x + FIELD_LEVEL_SETTINGS[level_settings[i][0]] / 2
                setattr(
                    self,
                    f"goal_red_{i}_{j}",
                    AssetBaseCfg(
                        prim_path=f"/Visuals/SoccerFieldVisuals/goal_red_{i}_{j}",
                        init_state=AssetBaseCfg.InitialStateCfg(
                            pos=(goal_red_x, sub_terrain_origin_y, 0.0), rot=(0.0, 0.0, 0.0, 1.0)
                        ),
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=MARL_ROOT_DIR + "/assets/goal/goal.usd",
                            scale=(0.6, float(0.6 * goal_scaler), 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                    ),
                )

                setattr(
                    self,
                    f"goal_blue_{i}_{j}",
                    AssetBaseCfg(
                        prim_path=f"/Visuals/SoccerFieldVisuals/goal_blue_{i}_{j}",
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(goal_blue_x, sub_terrain_origin_y, 0.0)),
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=MARL_ROOT_DIR + "/assets/goal/goal.usd",
                            scale=(0.6, 0.6 * goal_scaler, 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                        ),
                    ),
                )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    base_movement = mdp.ActionTermCfg(class_type=BaseMovementAction, asset_name="agents")
    ball_control = mdp.ActionTermCfg(class_type=BallControlAction, asset_name="ball")


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material_ball = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=".*"),
            "static_friction_range": (0.1, 0.5),
            "dynamic_friction_range": (0.1, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class WorldStateObservations(ObsGroup):
        concatenate_terms: bool = False
        world_state = ObsTerm(func=O.world_state, noise=Unoise(n_min=-0.0, n_max=0.0))

    @configclass
    class BotObservations(ObsGroup):
        concatenate_terms: bool = False
        base_own_pose_w = ObsTerm(
            func=O.base_own_pose_w,
            params={"normalize": False, "expand_rot": False},
        )
        ball_pos_b = ObsTerm(func=O.ball_pos_b)
        ball_pos_w = ObsTerm(func=O.ball_pos_w, history_length=2, params={"normalize": False})
        closest_agent_to_ball = ObsTerm(func=O.closest_agent_to_ball)
        player_id = ObsTerm(func=O.player_id)
        field_infos = ObsTerm(func=O.field_infos, params={"normalize": False})

    @configclass
    class NeighborStates(ObsGroup):
        concatenate_terms: bool = True  # turns off the noise in all observations
        teammate_pose = ObsTerm(
            func=O.closest_teammate_pose,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=2,
            params={"normalize": WORLD_POS_NORMALIZATION, "expand_rot": True, "max_num_neighbor": 5},
        )

        opponent_pose = ObsTerm(
            func=O.closest_opponent_pose,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=2,
            params={"normalize": WORLD_POS_NORMALIZATION, "expand_rot": True, "max_num_neighbor": 5},
        )

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        concatenate_terms = True
        # observation terms (order preserved)
        base_own_vel = ObsTerm(func=O.base_own_vel, noise=Unoise(n_min=-0.005, n_max=0.005))
        base_own_pose_w = ObsTerm(
            func=O.base_own_pose_w,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            params={"normalize": True, "expand_rot": True},
        )
        ball_vel_w = ObsTerm(func=O.ball_vel_w, noise=Unoise(n_min=-0.005, n_max=0.005))
        ball_pos_w = ObsTerm(func=O.ball_pos_w, noise=Unoise(n_min=-0.002, n_max=0.002), params={"normalize": True})
        field_infos = ObsTerm(func=O.field_infos, noise=Unoise(n_min=-0.0, n_max=0.0), params={"normalize": True})

    # observation groups
    policy: PolicyCfg = PolicyCfg(enable_corruption=True)
    critic: PolicyCfg = PolicyCfg(enable_corruption=False)
    neighbor = NeighborStates(enable_corruption=False)
    neighbor_critic = NeighborStates(enable_corruption=False)

    world_state = WorldStateObservations()
    bots = BotObservations()


@configclass
class RewardsCfg:
    @configclass
    class CustomRewTerm(RewTerm):
        decay_condition_type: str = None
        decay_threshhold: float = 0

    """Reward terms for the MDP."""

    score = RewTerm(func=R.score, weight=100.0)
    ball_outside_field_penalty = RewTerm(func=R.ball_outside_field_penalty, weight=1.0)
    collision_penalty = RewTerm(
        func=R.collision_penalty, weight=1.0, params={"collision_radius_agent": 0.2, "collision_radius_goal_post": 0.17}
    )

    vel_2_ball = CustomRewTerm(
        func=R.vel_2_ball,
        weight=0.5,
        decay_condition_type="score",
        decay_threshhold=1.5,
        params={"mode": "until_close"},
    )
    vel_ball_2_goal = CustomRewTerm(
        func=R.vel_ball_2_goal,
        weight=2.0,
        decay_condition_type="score",
        decay_threshhold=1.5,
    )
    ball_direction = CustomRewTerm(
        func=R.ball_direction,
        weight=0.5,
        decay_condition_type="score",
        decay_threshhold=1.5,
        params={"std": 0.4},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_scored = DoneTerm(func=T.goal_scored, time_out=False)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.field_levels)


@configclass
class SoccerMARLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    soccer_game = SoccerGameCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.02
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.scene.ball = BallCfg(prim_path="{ENV_REGEX_NS}/Ball")
        self.create_agent()
        if self.soccer_game.create_field_visuals:
            self.scene.create_soccer_field_visual()

    def create_agent(self):
        for i in range(self.soccer_game.num_agents_per_team):
            self.scene.agents.rigid_objects[f"{i}"] = AgentCfg(
                prim_path="{ENV_REGEX_NS}" + f"/agent_blue_{i}",
                base_color=(0.0, 0.0, 1.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4 * (i + 1), -1.0, 0.08)),
            )
        for i in range(self.soccer_game.num_agents_per_team):
            self.scene.agents.rigid_objects[f"{i+self.soccer_game.num_agents_per_team}"] = AgentCfg(
                prim_path="{ENV_REGEX_NS}" + f"/agent_red_{i}",
                base_color=(1.0, 0.0, 0.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4 * (i + 1), -1.0, 0.08)),
            )


@configclass
class SoccerMARLEnvPlayCfg(SoccerMARLEnvCfg):
    soccer_game = SoccerGameCfg(
        num_agents_per_team=5,
        num_agents_range=[5, 5],
        eval_env_ratio=0.0,
        level_settings=[[3, 3]],
        create_field_visuals=True,
    )
    scene = MySceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        self.scene.terrain.terrain_generator.sub_terrains["soccer_field"].level_settings = (
            self.soccer_game.level_settings
        )
        self.episode_length_s = 120
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.terrain_generator.num_rows = len(self.soccer_game.level_settings)
        self.scene.__post_init__()
        super().__post_init__()
