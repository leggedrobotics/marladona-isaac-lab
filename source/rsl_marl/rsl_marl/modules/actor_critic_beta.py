# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

import torch
import torch.nn as nn
from torch.distributions import Beta

from rsl_marl.modules.neighbor_net import NeighborNet


class ActorCriticBeta(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_neighbor_obs,
        max_num_teammate,
        max_num_opponent,
        num_actors,
        num_actions,
        global_encoding_dim=16,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticBeta.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        self.num_actions = num_actions
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        self.num_neighbor_obs = num_neighbor_obs

        self.total_num_neighbor_obs = num_neighbor_obs * (max_num_teammate + max_num_opponent)
        self.preprocess_net = NeighborNet(
            num_neighbor_obs, mlp_input_dim_a, max_num_teammate, max_num_opponent, global_encoding_dim
        )
        actor_critic_input_dim = mlp_input_dim_a + global_encoding_dim * 2

        # Policy
        actor_layers = []

        actor_layers.append(nn.Linear(actor_critic_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions * 2))
                actor_layers.append(nn.Softplus(beta=1))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(actor_critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.preprocess_net}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.distribution = None

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def action_alpha(self):
        return self.distribution.concentration1

    @property
    def action_beta(self):
        return self.distribution.concentration0

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_beta_parameters(self, params):
        alpha = params[:, : self.num_actions]
        beta = params[:, self.num_actions :]
        return alpha, beta

    def update_distribution(self, observations):
        params = self.actor(observations)
        alpha, beta = self.get_beta_parameters(params)
        self.distribution = Beta(alpha, beta)

    def act(self, observations, **kwargs):
        ego_obs = observations[:, : -self.total_num_neighbor_obs]
        neighbor_obs = observations[:, -self.total_num_neighbor_obs :]

        neighbor_feats = self.preprocess_net(ego_obs, neighbor_obs)
        full_observations = torch.cat([ego_obs, neighbor_feats], dim=1)
        self.update_distribution(full_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        ego_obs = observations[:, : -self.total_num_neighbor_obs]
        neighbor_obs = observations[:, -self.total_num_neighbor_obs :]

        neighbor_feats = self.preprocess_net(ego_obs, neighbor_obs)
        full_observations = torch.cat([ego_obs, neighbor_feats], dim=1)
        params = self.actor(full_observations)
        alpha, beta = self.get_beta_parameters(params)
        actions_mean = alpha / (alpha + beta)
        return (actions_mean - 0.5) * 2.0

    def evaluate(self, critic_observations, **kwargs):
        critic_obs = critic_observations[:, : -self.total_num_neighbor_obs]
        neighbor_obs = critic_observations[:, -self.total_num_neighbor_obs :]

        neighbor_feats = self.preprocess_net(critic_obs, neighbor_obs)
        full_observations = torch.cat([critic_obs, neighbor_feats], dim=1)
        value = self.critic(full_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
