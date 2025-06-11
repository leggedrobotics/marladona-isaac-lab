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
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.log

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_base import ManagerBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab_marl.assets.env_data import SoccerAgentsCfg
    from isaaclab_marl.tasks.soccer.soccer_marl_env import SoccerMARLEnv
    from isaaclab_marl.assets.agents import Agents

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.math import quat_rotate_inverse

from isaaclab_marl.utils.math_utils import rotate2d


class MultiAgentAction:
    @property
    def num_agents_per_team(self) -> int:
        """Number of environments."""
        return self._env.env_data.num_agents_per_team

    @property
    def num_agents_per_env(self) -> int:
        """Number of environments."""
        return self._env.env_data.num_agents_per_team * 2


class BaseMovementAction(ActionTerm, MultiAgentAction):

    cfg: ActionTermCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    agents: Agents
    agent_cfg: SoccerAgentsCfg

    def __init__(self, cfg: ActionTermCfg, env: SoccerMARLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.agents = env.agents
        self.agent_cfg = env.agents.agent_cfg
        # # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.num_agents_per_env, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        self.target_velocity_cmd = torch.zeros_like(self._processed_actions[..., :3])
        # self.target_force_cmd = torch.zeros_like(self.target_velocity_cmd)

        self._force = torch.zeros(self.num_envs, self.num_agents_per_env, 3, device=self.device)
        self._torque = torch.zeros(self.num_envs, self.num_agents_per_env, 3, device=self.device)

        self._scale = 1.0
        self._offset = 0.0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.view(self.num_envs, self.num_agents_per_env, -1)
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self):
        self.compute()

        self._asset.set_external_force_and_torque(self._force, self._torque)

    def compute(self):
        actions = self._processed_actions
        original_target_vel = torch.clip(actions[..., :2], -1, 1)
        target_vel = original_target_vel.clone()

        target_vel[..., 0] = original_target_vel[..., 0] * torch.sqrt(1 - original_target_vel[..., 1] ** 2 / 2)
        target_vel[..., 1] = original_target_vel[..., 1] * torch.sqrt(1 - original_target_vel[..., 0] ** 2 / 2)

        target_rot_vel = torch.clip(actions[..., 2], -1, 1)

        speed_limit = self.agent_cfg.cmd_speed_limit

        target_vel[..., 0][target_vel[..., 0] > 0] = target_vel[..., 0][target_vel[..., 0] > 0] * speed_limit[0][0]
        target_vel[..., 0][target_vel[..., 0] < 0] = target_vel[..., 0][target_vel[..., 0] < 0] * (-speed_limit[0][1])
        target_vel[..., 1] = target_vel[..., 1] * speed_limit[1][0]
        target_rot_vel = target_rot_vel * speed_limit[2][0]

        self.target_velocity_cmd[..., :2] = target_vel
        self.target_velocity_cmd[..., 2] = target_rot_vel

        self._force[..., :2] = (target_vel - self.agents.base_lin_vel_b) * self.agent_cfg.translation_force_d_gain
        self._torque[..., 2] = (target_rot_vel - self.agents.base_ang_vel_w) * self.agent_cfg.rotation_torque_d_gain

        inactive_agents = self._env.env_data.inactive.view(self.num_envs, -1)
        self._force[inactive_agents] = 0
        self._torque[inactive_agents] = 0


class BallControlAction(ActionTerm, MultiAgentAction):
    cfg: ActionTermCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: ActionTermCfg, env: SoccerMARLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.ball = env.ball
        self.agents = env.agents
        self.ball_cfg = env.ball.ball_cfg
        # # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.num_agents_per_env, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._scale = 1.0
        self._offset = 0.0

        self.total_target_cmd_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._force = torch.zeros(self.num_envs, 3, device=self.device)
        self._torque = torch.zeros(self.num_envs, 3, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def env_data(self) -> torch.Tensor:
        return self._env.env_data

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.view(self.num_envs, self.num_agents_per_env, -1)
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self):
        self.compute()
        self._asset.set_external_force_and_torque(self._force.unsqueeze(1), self._torque.unsqueeze(1))

    def compute(self):
        def expand(env_base_tensor):
            return self._env.env_data.expand_to_env(env_base_tensor, flatten_type="env")

        def expand_vector(env_base_tensor):
            return self._env.env_data.expand_vector_to_env(env_base_tensor, flatten_type="env")

        actions = self._processed_actions
        offset = 0.01
        target_velocity_cmd_b = torch.zeros_like(actions)
        target_velocity_cmd_b[..., 0] = actions[..., 0] * torch.sqrt(1 - actions[..., 1] ** 2 / 2)
        target_velocity_cmd_b[..., 1] = actions[..., 1] * torch.sqrt(1 - actions[..., 0] ** 2 / 2)
        target_velocity_cmd_b[..., 0] += offset
        target_velocity_cmd_b[..., 0] /= 2

        target_velocity_cmd_w = rotate2d(
            target_velocity_cmd_b[..., 0].flatten(),
            target_velocity_cmd_b[..., 1].flatten(),
            -self.agents.base_pose_w[..., 2].flatten(),
        )

        ball_pos_b = self.agents.ball_pos_b
        ball_controllable = torch.logical_and(
            torch.logical_and(
                torch.logical_and(ball_pos_b.norm(dim=-1) < self.ball_cfg.kick_margin, ball_pos_b[:, 0] > 0),
                torch.abs(ball_pos_b[:, 1]) < 0.1,
            ),
            expand(self._asset.data.root_pos_w[:, 2]) < 0.08,
        ).unsqueeze(-1)

        target_velocity_cmd_w *= ball_controllable

        target_force_cmd_w = (
            (target_velocity_cmd_w - expand_vector(self.ball.base_lin_vel_w))
            * self.ball_cfg.kick_force_gain
            * ball_controllable
        )

        inactive_agent = self._env.env_data.inactive.flatten()

        target_velocity_cmd_w[inactive_agent] = 0.0
        target_force_cmd_w[inactive_agent] = 0.0
        ball_controllable[inactive_agent] = False

        # total_target_cmd_w = torch.mean(target_velocity_cmd_w.view(self.env_data.team_flatten_pose_shape), dim=1)
        self.total_target_cmd_w[:, :2] = torch.sum(target_force_cmd_w.view(self.env_data.team_flatten_pos_shape), dim=1)

        ball_state_modifyable = torch.any(ball_controllable.view(self.env_data.team_flatten_base_shape), dim=1)
        self._force[:] = quat_rotate_inverse(self._asset.data.root_quat_w, self.total_target_cmd_w)

        # passive physic: from environment friction
        self._torque[:] = -self._asset.data.root_ang_vel_b * self.ball.rolling_friction


class MultiAgentActionManager(ManagerBase, MultiAgentAction):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        # check if config is None
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # create buffers to store actions
        self._action = torch.zeros((self.num_envs * self.num_agents_per_env, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        idx = 0
        for name, term in self._terms.items():
            term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
            terms.append((name, term_actions.tolist()))
            idx += term.action_dim
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term
