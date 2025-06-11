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

import os
import torch
from copy import deepcopy

from rsl_marl.modules import ActorCriticBeta

from isaaclab_marl.config import WKS_LOGS_DIR


class PolicyReplayManager:
    is_recurrent = False
    policy_buffer = dict()

    def __init__(self, cfg, actor_critic, num_envs, num_replay_agents_per_env, device):
        self.cfg = cfg
        self.initial_policy = deepcopy(actor_critic)
        self.device = device
        self.actor_critic = actor_critic

        self.num_envs = num_envs
        self.num_replay_agents_per_env = num_replay_agents_per_env

        self.max_num_levels = min(self.cfg["max_num_policy_replay_level"], self.num_envs)
        if self.cfg["dynamic_generate_replay_level"]:
            self.replay_from_previous_experiments = False
            self.num_levels = 0
        elif self.cfg["load_path"] == "":
            self.replay_from_previous_experiments = False
            self.num_levels = 0
        else:
            self.replay_from_previous_experiments = True

        self.next_policy_level_index = 0
        if self.replay_from_previous_experiments:
            log_folder = os.path.join(WKS_LOGS_DIR, cfg["load_path"])
            self.load_buffer_policy(log_folder)
        else:
            for _ in range(4):
                self.append_policy(self.initial_policy)

        if self.num_levels > 0:
            self.generate_env_level_indices()

    def generate_env_level_indices(self):
        if self.cfg["shuffle"]:
            env_indices = torch.randperm(self.num_envs, dtype=torch.int32, device=self.device)
        else:
            env_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.policy_level_index_dict = dict()
        env_indices_split = torch.chunk(env_indices, self.num_levels)
        for i, key in enumerate(self.policy_buffer.keys()):
            self.policy_level_index_dict[key] = env_indices_split[i]

    def load_buffer_policy(self, log_folder):
        models = [file for file in os.listdir(log_folder) if "model" in file]
        models.sort(key=lambda m: f"{m:0>15}")

        self.num_levels = min(self.max_num_levels, len(models))
        pathes = [os.path.join(log_folder, model) for model in models][-self.num_levels :]

        for i, path in enumerate(pathes):
            loaded_dict = torch.load(path)
            self.policy_buffer[f"level_{i}"] = deepcopy(self.initial_policy)
            self.policy_buffer[f"level_{i}"].load_state_dict(loaded_dict["model_state_dict"])
        self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.next_policy_level_index = len(pathes)

    def append_policy(self, policy: ActorCriticBeta):
        self.policy_buffer[f"level_{self.next_policy_level_index}"] = deepcopy(policy)
        if self.num_levels < self.max_num_levels:
            self.num_levels += 1
        else:
            del self.policy_buffer[f"level_{self.next_policy_level_index - self.num_levels}"]
        self.next_policy_level_index += 1
        self.generate_env_level_indices()

    def act(self, obs, level=0):
        self.initial_policy.eval()
        if level == 0:
            actions = self.initial_policy.act_inference(obs)
        elif level == -1:
            actions = self.actor_critic.act_inference(obs)
        else:
            actions = self.initial_policy.act_inference(obs)
            for level_key, policy in self.policy_buffer.items():
                policy.eval()
                env_indices = self.policy_level_index_dict[level_key]
                actions[env_indices] = policy.act_inference(obs[env_indices])
        return actions
