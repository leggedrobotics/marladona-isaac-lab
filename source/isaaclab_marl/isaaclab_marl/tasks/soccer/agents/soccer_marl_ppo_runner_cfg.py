from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PolicyReplayCfg:
    dynamic_generate_replay_level = True
    shuffle = False
    score_difference = 0.75
    load_path: str = ""
    max_num_policy_replay_level = 2


@configclass
class SymmetryCfg:
    use_symmetry = True
    # decide what to use
    use_augmentation: bool = True  # this adds symmetric trajectories to the batch
    use_loss: bool = False  # this adds symmetry loss term to the loss function
    # symmetry params
    # coefficient for symmetry loss term
    # if 0, then no symmetry loss is used
    symmetry_coeff = 0.0


@configclass
class CustomRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: SymmetryCfg = SymmetryCfg()


@configclass
class SoccerMARLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 10000
    save_interval = {"0": 20, "200": 50, "500": 100, "2000": 500}
    experiment_name = "marl_soccer"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticBeta",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = CustomRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    policy_replay: PolicyReplayCfg = PolicyReplayCfg()
