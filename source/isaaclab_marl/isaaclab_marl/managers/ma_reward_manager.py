# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

"""Reward manager for computing reward signals for a given world."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.reward_manager import RewardManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MultiAgentRewardManager(RewardManager):
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env: The environment instance.
        """

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # prepare extra info to store individual reward term information
        self.performance_metrics = {"average_score": 0}

        # Overwrite and reinitialize the following members.
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(
                self.num_envs * self.num_agents_per_env, dtype=torch.float, device=self.device
            )
        # create buffer for managing reward per environment
        self._reward_buf = torch.zeros(self.num_envs * self.num_agents_per_env, dtype=torch.float, device=self.device)

        # Buffer which stores the current step reward for each term for each environment
        self._step_reward = torch.zeros(
            (self.num_envs * self.num_agents_per_env, len(self._term_names)), dtype=torch.float, device=self.device
        )

    def compute(self, dt):
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            if hasattr(term_cfg, "decay_condition_type") and hasattr(term_cfg, "decay_threshhold"):
                if term_cfg.decay_threshhold < self.performance_metrics["average_score"]:
                    term_cfg.weight = 0
        return super().compute(dt)

    """
    Properties.
    """

    @property
    def num_agents_per_env(self) -> int:
        """Number of environments."""
        return self._env.env_data.num_agents_per_team * 2
