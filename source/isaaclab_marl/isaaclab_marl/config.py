import numpy as np
import os

WORKSPACE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
MARL_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

SCALING = 0.6
FIELD_LEVEL_SETTINGS = ((np.arange(4) + 2) * 0.2 * 9.0 * SCALING).tolist()
GOAL_LEVEL_SETTINGS = (np.array([6.0, 4.6, 3.4, 2.4, 1.6]) * SCALING).tolist()
FIELD_WIDTH = 6.0 * SCALING

APPLY_RANDOMIZATION = True
WORLD_POS_NORMALIZATION = True

POLICY_CLASS_NAME = "ActorCriticBeta"  # "ActorCriticBeta" or "ActorCritic"

TRAINING_MODUS = "selfplay"  # "none" or "mirror" or "selfplay" or "bot
TERMINATION = True

SIMULATION2D_SCALER = 19
SIMULATION2D_TIME_FACTOR = 10.0
BALL_MASS = 0.044
BALL_MAX_SPEED = 1.5 / SIMULATION2D_SCALER * SIMULATION2D_TIME_FACTOR
KICK_MODEL = "default"  # "default" or "simulation2d"
MIRROR_BALL_INIT_POS = False
PREPROCESS_NET_TYPE = "neighbor_net"  # "attention" or "neighbor_net"
