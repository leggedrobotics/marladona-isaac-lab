# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

import atexit
import numpy as np
import statistics

# python
import time

# torch
import torch
from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING

from rsl_marl.algorithms import PPO
from rsl_marl.modules import ActorCriticBeta
from rsl_marl.modules.bots import Bots
from rsl_marl.modules.policy_replay_manager import PolicyReplayManager
from rsl_marl.utils.log_manager import LogManager

# rsl-rl
from rsl_rl.env import VecEnv

if TYPE_CHECKING:
    from isaaclab_marl.assets.env_data import EnvData
    from isaaclab_marl.tasks.soccer.soccer_marl_env_cfg import SoccerMARLEnvCfg


class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu", command_args=None):
        self.env_cfg: SoccerMARLEnvCfg = env.cfg
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.policy_replay_cfg = train_cfg["policy_replay"]
        self.device = device
        self.env = env
        self.env_data: EnvData = self.env.unwrapped.env_data

        obs_dict = self.env.get_observations()
        self.num_obs = obs_dict["policy"].shape[1]
        if "critic" in obs_dict:
            self.num_critic_obs = obs_dict["critic"].shape[1]
        else:
            self.num_critic_obs = self.num_obs

        self.max_num_teammate = min(
            self.env_cfg.soccer_game.num_agents_per_team - 1,
            self.env_cfg.observations.neighbor.teammate_pose.params["max_num_neighbor"],
        )
        self.max_num_opponent = min(
            self.env_cfg.soccer_game.num_agents_per_team,
            self.env_cfg.observations.neighbor.teammate_pose.params["max_num_neighbor"],
        )
        self.max_num_obs_agents = self.max_num_teammate + self.max_num_opponent

        self.num_neighbor_obs = obs_dict["neighbor"].shape[-1] // self.max_num_obs_agents

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCriticBeta = actor_critic_class(
            self.num_obs,
            self.num_critic_obs,
            self.num_neighbor_obs,
            self.max_num_teammate,
            self.max_num_opponent,
            self.max_num_obs_agents,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.alg.obs_mirror_index = torch.tensor(
            self.env.unwrapped.observation_manager._group_symmetry_index["policy"]
            + [self.num_obs + i for i in self.env.unwrapped.observation_manager._group_symmetry_index["neighbor"]]
        ).to(self.device)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.bot_policy = Bots(self.env, self.device)

        self.total_num_replay_actors = len(self.env_data.policy_replay_actor_index.nonzero())

        self.logger = LogManager(
            self.env.unwrapped, self.alg, log_dir, command_args, self.env_cfg, train_cfg, device=self.device
        )

        if self.env_data.num_policy_replay_agents_per_env > 0:
            self.policy_replay_manager = PolicyReplayManager(
                self.policy_replay_cfg,
                actor_critic,
                self.total_num_replay_actors,
                self.env_data.num_policy_replay_agents_per_env,
                self.device,
            )
        self.reset_buffers()

        # Log
        self.log_dir = log_dir
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.epoch_counter = 0

    def reset_buffers(self):
        if self.env_data.num_policy_replay_agents_per_env > 0:
            self.policy_replay_manager.num_envs = len(self.env_data.policy_replay_actor_index.nonzero())
        self.total_num_training_actors = len(self.env_data.training_actor_index.nonzero())
        self.logger.init_config(self.total_num_training_actors)
        # init storage and model
        self.alg.init_storage(
            self.total_num_training_actors,
            self.num_steps_per_env,
            [self.num_obs + self.num_neighbor_obs * self.max_num_obs_agents],
            [self.num_critic_obs + self.num_neighbor_obs * self.max_num_obs_agents],
            [self.env.num_actions],
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        atexit.register(self.exit_handler)

        self.logger.init_writer()
        self.logger.init_trajectory_buffer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs_dict = self.env.get_observations()

        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []

        scorebuffer_length = 250
        scorebufferBlue = deque(maxlen=scorebuffer_length)
        scorebufferRed = deque(maxlen=scorebuffer_length)

        scorebufferBlueEval = deque(maxlen=scorebuffer_length)
        scorebufferRedEval = deque(maxlen=scorebuffer_length)
        scorebufferDiff = deque(maxlen=scorebuffer_length)

        metric_name_list = []
        metric_name_list = ["reward", "episode_length"]

        metric_buffer_dict = {}
        metric_buffer_dict_eval = {}
        metric_sum_dict = {}

        for metric_name in metric_name_list:
            metric_buffer_dict[metric_name] = deque(maxlen=100)
            metric_buffer_dict_eval[metric_name] = deque(maxlen=100)
            metric_sum_dict[metric_name] = torch.zeros(
                self.env.num_envs * self.env_data.num_agents_per_env,
                dtype=torch.float,
                device=self.device,
            )

        cur_level = torch.zeros(self.env.num_envs, dtype=torch.int64, device=self.device)
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        init_pos_level = 0

        last_save_curriculum_level = 0

        for it in range(start_iter, tot_iter):
            self.epoch_counter = it
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    obs = torch.cat([obs_dict["policy"], obs_dict["neighbor"]], dim=1)
                    critic_obs = torch.cat([obs_dict["critic"], obs_dict["neighbor_critic"]], dim=1)

                    actions = torch.zeros(
                        self.env.num_envs * self.env_data.num_agents_per_env,
                        self.env.num_actions,
                        device=self.device,
                    )
                    actions[self.env_data.training_actor_index] = self.alg.act(
                        obs[self.env_data.training_actor_index], critic_obs[self.env_data.training_actor_index]
                    )
                    actions[self.env_data.eval_actor_index] = self.alg.actor_critic.act_inference(
                        obs[self.env_data.eval_actor_index]
                    )
                    if self.env_data.num_policy_replay_agents_per_env > 0:
                        actions[self.env_data.policy_replay_actor_index] = self.policy_replay_manager.act(
                            obs[self.env_data.policy_replay_actor_index], level=8
                        )
                    actions[self.env_data.bots_actor_index] = self.bot_policy.act(
                        obs_dict["bots"],
                        actions,
                        self.env_data.bots_actor_index,
                        configs=["keeper"] + ["player"] * (self.env_data.num_agents_per_team - 1),
                    )

                    obs_dict, rewards, actor_dones, infos = self.env.step(actions)
                    env_dones = actor_dones.reshape(self.env_data.team_flatten_base_shape)[:, 0]

                    rewards_clone = rewards.clone()

                    training_info = deepcopy(infos)
                    training_info["time_outs"] = training_info["time_outs"][self.env_data.training_actor_index]
                    self.alg.process_env_step(
                        rewards_clone[self.env_data.training_actor_index],
                        actor_dones[self.env_data.training_actor_index],
                        training_info,
                    )

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])

                        update_log_index = torch.logical_or(
                            self.env_data.training_actor_index, self.env_data.eval_actor_index
                        )
                        metric_sum_dict["reward"][update_log_index] += (rewards)[update_log_index]
                        metric_sum_dict["episode_length"][update_log_index] += 1

                        new_actor_ids_train = (
                            torch.logical_and((actor_dones > 0), self.env_data.training_actor_index)
                            .nonzero(as_tuple=False)
                            .flatten()
                        )
                        new_actor_ids_eval = (
                            torch.logical_and((actor_dones > 0), self.env_data.eval_actor_index)
                            .nonzero(as_tuple=False)
                            .flatten()
                        )

                        for metric_name in metric_name_list:
                            if len(new_actor_ids_train) != 0:
                                metric_buffer_dict[metric_name].extend(
                                    metric_sum_dict[metric_name][new_actor_ids_train].cpu().numpy().tolist()
                                )
                            if len(new_actor_ids_eval) != 0:
                                metric_buffer_dict_eval[metric_name].extend(
                                    metric_sum_dict[metric_name][new_actor_ids_eval].cpu().numpy().tolist()
                                )
                            if len(new_actor_ids_train) != 0:
                                metric_sum_dict[metric_name][new_actor_ids_train] = 0
                            if len(new_actor_ids_eval) != 0:
                                metric_sum_dict[metric_name][new_actor_ids_eval] = 0

                        new_env_ids_train = (
                            torch.logical_and((env_dones > 0), self.env_data.is_training_env)
                            .nonzero(as_tuple=False)
                            .flatten()
                        )
                        new_env_ids_eval = (
                            torch.logical_and((env_dones > 0), torch.logical_not(self.env_data.is_training_env))
                            .nonzero(as_tuple=False)
                            .flatten()
                        )
                        self.logger.update_trajectory_buffer(
                            env_dones[torch.logical_not(self.env_data.is_training_env)] > 0,
                            infos["score"][torch.logical_not(self.env_data.is_training_env)],
                        )

                        scorebufferBlue.extend(infos["score"][new_env_ids_train][:, 0].cpu().numpy().tolist())
                        scorebufferRed.extend(infos["score"][new_env_ids_train][:, 1].cpu().numpy().tolist())

                        scorebufferBlueEval.extend(infos["score"][new_env_ids_eval][:, 0].cpu().numpy().tolist())
                        scorebufferRedEval.extend(infos["score"][new_env_ids_eval][:, 1].cpu().numpy().tolist())

                        scorebufferDiff.extend(
                            (infos["score"][new_env_ids_train][:, 0] - infos["score"][new_env_ids_train][:, 1])
                            .cpu()
                            .numpy()
                            .tolist()
                        )

                        cur_level[:] = self.env_data.field_curriculum_level[:]

                stop = time.time()
                collection_time = stop - start
                action_log = actions
                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs[self.env_data.training_actor_index])

            policy_good_enough = (
                len(scorebufferDiff) != 0
                and statistics.mean(scorebufferDiff) > self.policy_replay_cfg["score_difference"]
                and it - last_save_curriculum_level > 100
            )

            self.env.unwrapped.init_pos_manager.current_level = init_pos_level

            selfplay_mode = (
                self.policy_replay_cfg["dynamic_generate_replay_level"]
                and self.env_data.num_policy_replay_agents_per_env > 0
            )

            if policy_good_enough:
                self.env.unwrapped.reward_manager.performance_metrics["average_score"] = statistics.mean(
                    scorebufferDiff
                )

            if policy_good_enough:
                last_save_curriculum_level = it
                scorebufferDiff.clear()
                level = 0
                if self.env_data.num_policy_replay_agents_per_env > 0:
                    level = self.policy_replay_manager.next_policy_level_index

                if selfplay_mode:
                    self.policy_replay_manager.append_policy(self.alg.actor_critic)
                    # self.logger.save_eval_trajectory(it, "selfplay")
                    self.logger.save_curriculum_state(it, level)

            mean_value_loss, mean_surrogate_loss, entropy_loss = self.alg.update()

            if policy_good_enough:
                self.logger.save_eval_trajectory(it, "curriculum")
                self.logger.save_curriculum_state(it, level)

                self.env_data.regenerate_level_config()
                if init_at_random_ep_len:
                    self.env.episode_length_buf = torch.randint_like(
                        self.env.episode_length_buf, high=int(self.env.max_episode_length)
                    )
                self.reset_buffers()
                if self.env_data.num_policy_replay_agents_per_env > 0:
                    self.policy_replay_manager.generate_env_level_indices()
                obs_dict = self.env.get_observations()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                terrain_level = self.env_data.field_curriculum_level.cpu().numpy()
                self.logger.log(locals())

            epoch_list = np.array([int(value) for value in self.save_interval.keys()])
            epoch_list_index = np.argwhere(epoch_list <= it)[-1][0]

            if it % list(self.save_interval.values())[epoch_list_index] == 0:
                self.logger.save_model(it)
                self.logger.save_eval_trajectory(it)
            ep_infos.clear()

            if it == start_iter:
                self.logger.init_log_folder()

        self.logger.save_model(self.current_learning_iteration)
        self.logger.save_eval_trajectory(it)

    def exit_handler(self):
        print("Interrupt detected, exiting...")
        self.logger.save_model(self.epoch_counter)
        self.logger.save_eval_trajectory(self.epoch_counter)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        return policy

    def get_inference_critic(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        critic = self.alg.actor_critic.evaluate
        return critic

    def get_inference_replay_policies(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        policies = []
        if self.env_data.num_policy_replay_agents_per_env > 0:
            for key, policy in self.policy_replay_manager.policy_buffer.items():
                if device is not None:
                    self.policy_replay_manager.policy_buffer[key].to(device)
                policy = self.policy_replay_manager.policy_buffer[key].act_inference
                policies.append(policy)
            policies.append(self.policy_replay_manager.actor_critic.act_inference)
        return policies

    def train_mode(self):
        self.alg.actor_critic.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
