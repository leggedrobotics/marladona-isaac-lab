"""Locomotion environments with velocity-tracking commands.

These environments are based on the `legged_gym` environments provided by Rudin et al.

Reference:
    https://github.com/leggedrobotics/legged_gym
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# gym.register(
#     id="Soccer",
#     entry_point="isaaclab_marl.tasks.soccer:CartDoublePendulumEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": multi_agnet_env_cfg.CartDoublePendulumEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-Soccer-v0",
    entry_point=f"{__name__}.soccer_marl_env:SoccerMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.soccer_marl_env_cfg:SoccerMARLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.soccer_marl_ppo_runner_cfg:SoccerMARLPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Soccer-Play-v0",
    entry_point=f"{__name__}.soccer_marl_env:SoccerMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.soccer_marl_env_cfg:SoccerMARLEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.soccer_marl_ppo_runner_cfg:SoccerMARLPPORunnerCfg",
    },
)
