# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications copyright (c) 2025 Zichong Li, ETH Zurich

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_marl.modules import ActorCriticBeta
from rsl_marl.storage import RolloutStorage


class PPO:
    actor_critic: ActorCriticBeta

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        symmetry_cfg=None,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.symmetry_cfg = symmetry_cfg
        self.obs_mirror_index = None
        self.action_mirror_index = torch.tensor([1, 2, 4], device=self.device)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()

        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        if isinstance(self.actor_critic.distribution, torch.distributions.Beta):
            self.transition.action_alpha = self.actor_critic.action_alpha.detach()
            self.transition.action_beta = self.actor_critic.action_beta.detach()
        else:
            self.transition.action_alpha = torch.zeros_like(self.actor_critic.action_mean.detach())
            self.transition.action_beta = torch.zeros_like(self.actor_critic.action_mean.detach())

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        if isinstance(self.actor_critic, ActorCriticBeta):
            return (self.transition.actions - 0.5) * 2.0
        else:
            return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        entropy_loss = 0
        mean_symmetry_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            old_alpha_batch,
            old_beta_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 2
            # original batch size
            original_batch_size = obs_batch.shape[0]

            if self.symmetry_cfg["use_augmentation"] and self.symmetry_cfg["use_symmetry"]:
                obs_batch = obs_batch.repeat(num_aug, 1)
                obs_batch[original_batch_size:, self.obs_mirror_index] *= -1
                critic_obs_batch = critic_obs_batch.repeat(num_aug, 1)
                critic_obs_batch[original_batch_size:, self.obs_mirror_index] *= -1
                actions_batch = actions_batch.repeat(num_aug, 1)
                actions_batch[original_batch_size:, self.action_mirror_index] = torch.clip(
                    1.0 - actions_batch[original_batch_size:, self.action_mirror_index], 0.000001, 0.999999
                )
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            if isinstance(self.actor_critic.distribution, torch.distributions.Beta):
                alpha_batch = self.actor_critic.action_alpha[:original_batch_size]
                beta_batch = self.actor_critic.action_beta[:original_batch_size]
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if isinstance(self.actor_critic.distribution, torch.distributions.Normal):
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                    elif isinstance(self.actor_critic.distribution, torch.distributions.Beta):
                        kl = torch.sum(
                            torch.lgamma(old_alpha_batch + old_beta_batch)
                            + torch.lgamma(alpha_batch)
                            + torch.lgamma(beta_batch)
                            - torch.lgamma(alpha_batch + beta_batch)
                            - torch.lgamma(old_alpha_batch)
                            - torch.lgamma(old_beta_batch)
                            + (old_alpha_batch - alpha_batch)
                            * (
                                torch.special.digamma(old_alpha_batch)
                                - torch.special.digamma(old_alpha_batch + old_beta_batch)
                            )
                            + (old_beta_batch - beta_batch)
                            * (
                                torch.special.digamma(old_beta_batch)
                                - torch.special.digamma(old_alpha_batch + old_beta_batch)
                            ),
                            axis=-1,
                        )
                    else:
                        raise NotImplementedError("KL divergence not implemented for this distribution")
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            entropy_loss += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        entropy_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, entropy_loss
