import torch
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


class CustomVecEnvWrapper(RslRlVecEnvWrapper):
  def get_observations(self) -> tuple[torch.Tensor, dict]:
    """Returns the current observations of the environment."""
    return self.unwrapped.observation_manager.compute()
    
  def step(self, actions):
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    # compute dones for compatibility with RSL-RL
    dones = (terminated | truncated).to(dtype=torch.long)
    # move extra observations to the extras dict
    extras["observations"] = obs_dict
    # move time out information to the extras dict
    # this is only needed for infinite horizon tasks
    if not self.unwrapped.cfg.is_finite_horizon:
        extras["time_outs"] = truncated

    # return the step information
    return obs_dict, rew, dones, extras