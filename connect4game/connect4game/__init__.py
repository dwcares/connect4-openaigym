import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Connect4Game-v0',
    entry_point='connect4game.envs:Connect4GameEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)
